"""
EMA-model for any PyTorch model.
"""

import torch
from torch import nn

class AnyEMA:
    def __init__(self, named_parameters, scale: float = 1.0, capture=[]):
        self.ema_parameters = {}
        with torch.no_grad():
            for name, param in named_parameters:
                if param.requires_grad and (len(capture) == 0 or name in capture):
                    ema_param = param.data.clone().detach() * scale
                    self.ema_parameters[name] = ema_param
    
    def update(self, named_parameters, decay: float = 0.999):
        with torch.no_grad():
            for name, param in named_parameters:
                if param.requires_grad and name in self.ema_parameters:
                    ema_param = self.ema_parameters[name]
                    ema_param.mul_(decay).add_(param.data, alpha=1 - decay)

    def swap(self, named_parameters):
        with torch.no_grad():
            for name, param in named_parameters:
                if param.requires_grad and name in self.ema_parameters:
                    temp = param.data.clone()
                    param.data.copy_(self.ema_parameters[name])
                    self.ema_parameters[name].copy_(temp)