import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss as CELoss, MSELoss, KLDivLoss
from torch.nn.functional import linear, normalize
from configs.base import Config
from typing import Tuple, Any

class MeanSquaredError(MSELoss):
    """Rewrite MeanSquaredError to support init with kwargs"""
    def __init__(self, cfg: Config, **kwargs):
        super(MeanSquaredError, self).__init__(**kwargs)
        self.cfg = cfg
        self.reduction = cfg.fusion_reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(input, target)

class CrossEntropyLoss(CELoss):
    """Rewrite CrossEntropyLoss to support init with kwargs"""

    def __init__(self, cfg: Config, **kwargs):
        super(CrossEntropyLoss, self).__init__(**kwargs)
        self.cfg = cfg

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(input, target)


class Kullback_LeiblerDivergenceLoss(KLDivLoss):
    """Rewrite Kullback_LeiblerDivergenceLoss to support init with kwargs"""

    def __init__(self, cfg: Config, **kwargs):
        super(Kullback_LeiblerDivergenceLoss, self).__init__(**kwargs)
        self.cfg = cfg
        self.reduction = cfg.distil_reduction
        self.T = cfg.T

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = F.log_softmax(input / self.T, dim=-1)
        target = F.softmax(target / self.T, dim=-1)
        
        return super().forward(input, target) * (self.T**2)