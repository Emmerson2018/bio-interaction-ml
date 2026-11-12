import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_samples_per_class(dataset_root: str) -> List[int]:
    """Conta imagens por classe no diretório do dataset para computar a distribuição a priori."""
    root = Path(dataset_root)
    if not root.exists():
        return []
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    counts = []
    for c_dir in class_dirs:
        n = len(list(c_dir.glob('*.png'))) + len(list(c_dir.glob('*.jpg')))
        counts.append(max(1, n))
    return counts


class LogitAdjustedLoss(nn.Module):
    """
    Logit Adjustment (Balanced Softmax) para desbalanceamento de classes (Menon et al., ICLR 2021).
    Ajusta os logits pela probabilidade a priori da classe no dataset de treino:
        z'_i = z_i + tau * log(pi_i)
    onde pi_i = N_i / N_total.
    Em inferência, o modelo sem ajuste prediz a distribuição balanceada P(y|x).
    """

    def __init__(self, class_counts: List[int], tau: float = 1.0, label_smoothing: float = 0.0):
        super().__init__()
        self.tau = float(tau)
        self.label_smoothing = float(label_smoothing)
        total = sum(class_counts)
        priors = torch.tensor([c / total for c in class_counts], dtype=torch.float32)
        log_priors = torch.log(priors + 1e-12) * self.tau
        self.register_buffer('log_priors', log_priors)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        adjusted_logits = logits + self.log_priors.to(logits.device)
        return F.cross_entropy(adjusted_logits, target, label_smoothing=self.label_smoothing)


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., ICCV 2017) com label smoothing para focar em amostras difíceis.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha) if alpha is not None else None
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, target, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        return focal_loss.mean()


def build_loss(train_opt: dict, num_classes: int, train_root: Optional[str] = None) -> nn.Module:
    """Factory para funções de perda suportadas."""
    loss_opt = train_opt.get('loss', {})
    loss_type = loss_opt.get('type', 'CrossEntropyLoss')
    label_smoothing = float(train_opt.get('label_smoothing', loss_opt.get('label_smoothing', 0.1)))

    if loss_type in ('BalancedSoftmax', 'LogitAdjusted', 'LogitAdjustedLoss'):
        tau = float(loss_opt.get('tau', 1.0))
        class_counts = loss_opt.get('class_counts')
        if class_counts is None and train_root:
            class_counts = count_samples_per_class(train_root)
        if not class_counts or len(class_counts) != num_classes:
            # Fallback uniforme se não conseguir ler
            class_counts = [1] * num_classes
        print(f"[LossFactory] Balanced Softmax ativado com tau={tau} e contagens por classe: {class_counts}")
        return LogitAdjustedLoss(class_counts=class_counts, tau=tau, label_smoothing=label_smoothing)

    elif loss_type in ('FocalLoss', 'Focal'):
        gamma = float(loss_opt.get('gamma', 2.0))
        alpha = loss_opt.get('alpha', None)
        print(f"[LossFactory] Focal Loss ativado com gamma={gamma}, alpha={alpha}, label_smoothing={label_smoothing}")
        return FocalLoss(gamma=gamma, alpha=alpha, label_smoothing=label_smoothing)

    else:
        # Padrão: CrossEntropyLoss com Label Smoothing
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
