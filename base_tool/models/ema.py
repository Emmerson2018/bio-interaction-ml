import copy
import torch
import torch.nn as nn


class ModelEMA:
    """
    Exponential Moving Average (EMA) para suavização de pesos da rede neural.
    theta_ema = decay * theta_ema + (1 - decay) * theta.
    Proporciona estabilidade superior contra flutuações de batch e melhor generalização.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, device=None):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = float(decay)
        self.device = device
        if self.device:
            self.ema.to(self.device)
        for p in self.ema.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        model_sd = model.state_dict()
        for k, ema_v in self.ema.state_dict().items():
            if k in model_sd:
                model_v = model_sd[k]
                if self.device:
                    model_v = model_v.to(self.device)
                if ema_v.dtype.is_floating_point:
                    ema_v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)
                else:
                    ema_v.copy_(model_v)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict)
