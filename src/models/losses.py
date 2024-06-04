import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss as CELoss, MSELoss
from torch.nn.functional import linear, normalize
from configs.base import Config
from typing import Tuple, Any

class MeanSquaredError(MSELoss):
    # change reduction to sum
    def __init__(self, cfg: Config, **kwargs):
        super(MeanSquaredError, self).__init__(**kwargs)
        self.reduction = cfg.reduction
        self.cfg = cfg

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(input, target)

class CrossEntropyLoss(CELoss):
    """Rewrite CrossEntropyLoss to support init with kwargs"""

    def __init__(self, cfg: Config, **kwargs):
        super(CrossEntropyLoss, self).__init__(**kwargs)
        self.cfg = cfg

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out = input
        return super().forward(out, target)
