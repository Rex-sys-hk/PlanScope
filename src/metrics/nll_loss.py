from typing import Any, Callable, Dict, Optional

import torch
from torchmetrics import Metric
# from pytorch_wavelets import DWTForward, DWTInverse
from .utils import sort_predictions
import ptwt
import pywt

# TODO: DEBUG THIS FUNCTION (?)
def mvn_loss(pred_dist: torch.Tensor, traj_gt: torch.Tensor, masks: torch.Tensor):
    """
    Computes negative log likelihood of ground truth trajectory under a predictive distribution with a single mode,
    with a bivariate Gaussian distribution predicted at each time in the prediction horizon

    :param pred_dist: parameters of a bivariate Gaussian distribution, shape [batch_size, sequence_length, 5]
    :param traj_gt: ground truth trajectory, shape [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth, shape [batch_size, sequence_length]
    :return:
    """
    mu_x = pred_dist[:, :, 0]
    mu_y = pred_dist[:, :, 1]
    x = traj_gt[:, :, 0]
    y = traj_gt[:, :, 1]

    sig_x = pred_dist[:, :, 2]
    sig_y = pred_dist[:, :, 3]
    rho = pred_dist[:, :, 4]
    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)

    nll = 0.5 * torch.pow(ohr, 2) * \
        (torch.pow(sig_x, 2) * torch.pow(x - mu_x, 2) +
         torch.pow(sig_y, 2) * torch.pow(y - mu_y, 2) -
         2 * rho * torch.pow(sig_x, 1) * torch.pow(sig_y, 1) * (x - mu_x) * (y - mu_y))\
        - torch.log(sig_x * sig_y * ohr) + 1.8379

    nll[nll.isnan()] = 0
    nll[nll.isinf()] = 0

    nll = torch.sum(nll * (1 - masks), dim=1) / torch.sum((1 - masks), dim=1)
    # Note: Normalizing with torch.sum((1 - masks), dim=1) makes values somewhat comparable for trajectories of
    # different lengths

    return nll

class MVNLoss(torch.nn.Module):
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
        mul_ade_loss: list[str]=['phase_loss', 'scale_loss'],
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

    def compute_dis(self, outputs: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor]):
        # def print_keys(d:Dict, pfix=">> "):
        #     for k,v in d.items():
        #         print(pfix, k)
        #         if isinstance(v, dict):
        #             print_keys(v, pfix+">> ")
        # print_keys(data)
        error = torch.tensor(0.0, device=outputs["trajectory"].device)
        if 'mvn' not in outputs:
            return error

        self.mask = self.mask.to(outputs["trajectory"].device)
        b,r,m,t,dim = outputs["trajectory"].shape
        trajectories = outputs["trajectory"].reshape(b, r*m, t, dim)
        probabilities = outputs["probability"].reshape(b, r*m)
        mvn = outputs["mvn"].reshape(b, r*m, t, -1)

        
        pred, _ = sort_predictions(trajectories, probabilities, k=self.k)
        pred= pred[:,:self.k,:,:2]
        pred = pred.permute(0, 2, 1)
        mvn, _ = sort_predictions(mvn, probabilities, k=self.k)

        target = data['agent']['position'][:,0,:,:2]
        target = target.permute(0, 2, 1)

        ### NLL loss
        error += mvn_loss(torch.cat([pred, mvn], dim=-1), target, self.mask)
            
        return error

    def forward(self, outputs: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor]):
        if self.with_grad:
            return self.compute_dis(outputs, data)
        
        with torch.no_grad():
            return self.compute_dis(outputs, data)

