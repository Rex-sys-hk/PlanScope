from typing import Any, Callable, Dict, Optional

import torch
from torchmetrics import Metric
# from pytorch_wavelets import DWTForward, DWTInverse
from .utils import sort_predictions
import ptwt
import pywt


class mulADE(torch.nn.Module):
    """Multiple Scope Average Displacement Error
    mulADE: The average L2 distance between the best forecasted trajectory and the ground truth.
            The best here refers to the trajectory that has the minimum endpoint error.
    """

    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        k: int = 1,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        with_grad: bool = False,
        history_length: int = 21,
        whole_length: int = 101,
        mul_ade_loss: list[str]=['phase_loss', 'angle_loss', 'scale_loss', 'v_loss'],
        max_horizon: int = 10,
    ) -> None:
        super().__init__()
        self.k = k
        self.with_grad = with_grad
        self.sum=torch.tensor(0.0)
        self.count=torch.tensor(0)

        self.dt = 0.1
        self.history_length = history_length
        self.whole_length = whole_length
        self.wavelet = 'cgau1' # real 'gaus1', 'mexh', 'morl' # complex 'cgau1', 'cmor', 'fbsp', 'shan'
        self.scales = torch.exp(torch.arange(0, 6, 0.25))
        self.widths = 1.0 / pywt.scale2frequency(self.wavelet, self.scales)
        scales_num = self.widths.shape[0]
        # self.mask = torch.triu(torch.ones(scales_num+5, 80), diagonal=1
        #             )[:-5].bool().flip(-1)
        self.mask = torch.ones(scales_num, self.whole_length).bool()
        for r in range(scales_num):
            s_ind = torch.floor(self.widths[r]).int()
            s_ind = s_ind + self.history_length
            s_ind = torch.max(s_ind, torch.tensor(self.history_length))
            s_ind = torch.min(s_ind, torch.tensor(self.whole_length))
            self.mask[r, s_ind:] = False
        self.mask = self.mask.flip(0)
        self.mask = self.mask.unsqueeze(0)
        self.mul_ade_loss = mul_ade_loss
        self.max_horizon = max_horizon

    def compute_dis(self, outputs: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor]):
        # def print_keys(d:Dict, pfix=">> "):
        #     for k,v in d.items():
        #         print(pfix, k)
        #         if isinstance(v, dict):
        #             print_keys(v, pfix+">> ")
        # print_keys(data)
        error = torch.tensor(0.0, device=outputs["trajectory"].device)
        history = data['agent']['velocity'][:, 0, :self.history_length, :2]
        self.mask = self.mask.to(outputs["trajectory"].device)
        b,r,m,t,dim = outputs["trajectory"].shape
        trajectories = outputs["trajectory"].reshape(b, r*m, t, dim)
        probabilities = outputs["probability"].reshape(b, r*m)

        
        pred, _ = sort_predictions(trajectories, probabilities, k=self.k)
        pred = pred[:,0,:,-2:]
        pred = torch.cat([history, pred], dim=-2)
        pred = pred.permute(0, 2, 1)

        target = data['agent']['velocity'][:,0,:,:2]
        # target = torch.cat([history, target], dim=-2)
        target = target.permute(0, 2, 1)
        if 'v_loss' in self.mul_ade_loss:
            v_error = torch.norm(
                pred - target, p=2, dim=-1
            )
            error += v_error[...,self.history_length:].mean()

        pred_coeff,_ = ptwt.cwt(pred, self.widths, self.wavelet, sampling_period=self.dt)
        pred_coeff = pred_coeff.permute(1, 0, 3, 2)

        target_coeff,_ = ptwt.cwt(target, self.widths, self.wavelet, sampling_period=self.dt)
        target_coeff = target_coeff.permute(1, 0, 3, 2)
        if 'phase_loss' in self.mul_ade_loss:
            pred_coeff_angle = pred_coeff
            target_coeff_angle = target_coeff
            angle_error = torch.norm(
                pred_coeff_angle - target_coeff_angle, p=2, dim=-1
            )
            angle_error = angle_error*self.mask
            angle_error = angle_error.sum(-1)/self.mask.sum(-1)
            angle_error = angle_error.mean(-1)
            error += angle_error.mean()

        if 'angle_loss' in self.mul_ade_loss:
            pred_coeff_angle = torch.angle(pred_coeff)
            target_coeff_angle = torch.angle(target_coeff)
            angle_error = torch.norm(
                pred_coeff_angle - target_coeff_angle, p=2, dim=-1
            )
            angle_error = angle_error*self.mask
            angle_error = angle_error.sum(-1)/self.mask.sum(-1)
            angle_error = angle_error.mean(-1)
            error += angle_error.mean()

        if 'scale_loss' in self.mul_ade_loss:
            pred_coeff_real = torch.real(pred_coeff)
            target_coeff_real = torch.real(target_coeff)
            scale_error = torch.norm(
                pred_coeff_real - target_coeff_real, p=2, dim=-1
            )
            scale_error = scale_error*self.mask
            scale_error = scale_error.sum(-1)/self.mask.sum(-1)
            scale_error = scale_error.mean(-1)
            error += scale_error.mean()            

        # if 'detail_loss' in self.mul_ade_loss:
        details = outputs["details"]
        if not len(details) == 0:
            level = len(details)
            packet = ptwt.wavedec(target[...,self.history_length:], 'haar', level = level-1, mode = 'constant')
            for p, d in zip(packet, details):
                b, r, m, t, dim = d.shape
                d = d.reshape(b, r*m, t, dim)
                d, _ = sort_predictions(d, probabilities, k=self.k)
                d = d[:,0,:,:2]
                d = d.permute(0, 2, 1)
                interval = (self.whole_length - self.history_length) // p.shape[-1]
                d = d[...,::interval]
                error += torch.norm(d-p[...,:d.shape[-1]], p=2, dim=-2)[...,:self.max_horizon].mean()
            
        return error

    def forward(self, outputs: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor]):
        if self.with_grad:
            return self.compute_dis(outputs, data)
        
        with torch.no_grad():
            return self.compute_dis(outputs, data)

