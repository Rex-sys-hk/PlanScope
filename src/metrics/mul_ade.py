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
        mul_norm: bool = False,
        wtd_with_history: bool = False,
        learning_output: str = 'velocity', # or 'velocity'
        wavelet: str = ['cgau1', 'constant', 'bior1.3', 'constant'],
    ) -> None:
        super().__init__()
        self.k = k
        self.with_grad = with_grad
        self.sum=torch.tensor(0.0)
        self.count=torch.tensor(0)

        self.dt = 0.1
        self.history_length = history_length
        self.whole_length = whole_length
        self.wavelet = wavelet[0] # real 'gaus1', 'mexh', 'morl' # complex 'cgau1', 'cmor', 'fbsp', 'shan'
        self.edge_mode = wavelet[1] # 'constant'
        self.d_wavelet = wavelet[2] # 'haar'
        self.d_edge_mode = wavelet[3] # 'constant'
        self.scales = torch.pow(2, torch.arange(0, 5, 1)) # 0,1,2,3,4
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
        self.mul_norm = mul_norm
        self.wtd_with_history = wtd_with_history
        self.learning_output = learning_output

    def cwt_loss(self, pred, target, device, wavelet='cgau1', mode='constant'):
        error = torch.tensor(0.0).to(device)
        pred_coeff,_ = ptwt.cwt(pred, self.widths, wavelet, sampling_period=self.dt)
        pred_coeff = pred_coeff.permute(1, 0, 3, 2)

        target_coeff,_ = ptwt.cwt(target, self.widths, wavelet, sampling_period=self.dt)
        target_coeff = target_coeff.permute(1, 0, 3, 2)
        if 'phase_loss' in self.mul_ade_loss:
            pred_coeff_vec = pred_coeff
            target_coeff_vec = target_coeff
            phase_error = torch.norm(
                pred_coeff_vec - target_coeff_vec, p=2, dim=-1
            )
            phase_error = phase_error*self.mask
            phase_error = phase_error.sum(-1)/self.mask.sum(-1)
            phase_error = phase_error.mean(-1)
            error += phase_error.mean()

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
        return error

    def dwt_loss(self, details, target, probabilities, device, wavelet='haar', mode='constant'):
        effective_num = 0
        detail_loss = torch.tensor(0.0).to(device)
        level = len(details)
        # target = data['agent'][self.learning_output][:,0,:,:2]
        target = target.permute(0, 2, 1)
        if not self.wtd_with_history:
            target = target[...,self.history_length:]
        packet = ptwt.wavedec(target, wavelet, level = level-1, mode = mode)
        for p, d in zip(packet, details):
            b, r, m, t, dim = d.shape
            d = d.reshape(b, r*m, t, dim)
            d, _ = sort_predictions(d, probabilities, k=self.k)
            d = d[:,0,:,:2]
            d = d.permute(0, 2, 1)
            interval = round(self.whole_length / p.shape[-1])
            d = d[...,::interval]
            horizon = self.max_horizon
            if self.wtd_with_history:
                history_points = self.history_length // interval
                horizon += history_points
            e=torch.norm(d[...,:p.shape[-1]]-p[...,:d.shape[-1]], p=2, dim=-2)[...,:horizon]
            if self.mul_norm:
                effective_num += e.shape[-1]
                detail_loss += e.sum()
            else:
                effective_num += 1
                detail_loss += e.mean()
        return detail_loss/effective_num

    def compute_dis(self, outputs: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor]):
        # def print_keys(d:Dict, pfix=">> "):
        #     for k,v in d.items():
        #         print(pfix, k)
        #         if isinstance(v, dict):
        #             print_keys(v, pfix+">> ")
        # print_keys(data)
        error = torch.tensor(0.0, device=outputs["trajectory"].device)
        self.mask = self.mask.to(outputs["trajectory"].device)
        b,r,m,t,dim = outputs["trajectory"].shape
        trajectories = outputs["trajectory"].reshape(b, r*m, t, dim)
        probabilities = outputs["probability"].reshape(b, r*m)

        
        pred, _ = sort_predictions(trajectories, probabilities, k=self.k)

        p_pred = pred[:,0,:,:2]
        p_history = data['agent']['position'][:, 0, :self.history_length, :2]
        p_pred = torch.cat([p_history, p_pred], dim=-2)
        p_pred = p_pred.permute(0, 2, 1)
        p_target = data['agent']['position'][:,0,:,:2]
        p_target = p_target.permute(0, 2, 1)

        v_pred = pred[:,0,:,-2:]
        v_history = data['agent']['velocity'][:, 0, :self.history_length, :2]
        v_pred = torch.cat([v_history, v_pred], dim=-2)
        v_pred = v_pred.permute(0, 2, 1)
        v_target = data['agent']['velocity'][:,0,:,:2]
        v_target = v_target.permute(0, 2, 1)

        if 'v_loss' in self.mul_ade_loss:
            v_error = torch.norm(
                v_pred - v_target, p=2, dim=-1
            )
            error += v_error[...,self.history_length:].mean()

        if 'cwt_v_loss' in self.mul_ade_loss:
            error += self.cwt_loss(v_pred, v_target, 
                            outputs["trajectory"].device, self.wavelet, self.edge_mode)
   
        if 'cwt_p_loss' in self.mul_ade_loss:
            error += self.cwt_loss(p_pred, p_target, 
                            outputs["trajectory"].device, self.wavelet, self.edge_mode)

        # recursive head loss
        details = outputs["details"]
        if not len(details) == 0:
            error += self.dwt_loss(details, 
                                   data['agent'][self.learning_output][:,0,:,:2], 
                                   probabilities,
                                   outputs["trajectory"].device, 
                                   self.d_wavelet,
                                   self.d_edge_mode)
            
        return error

    def forward(self, outputs: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor]):
        if self.with_grad:
            return self.compute_dis(outputs, data)
        
        with torch.no_grad():
            return self.compute_dis(outputs, data)

