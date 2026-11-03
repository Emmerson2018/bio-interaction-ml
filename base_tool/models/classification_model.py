import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2

from base_tool.archs import build_network
from base_tool.models.base_model import BaseModel
from base_tool.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class ClassificationModel(BaseModel):
    def __init__(self, opt):
        super(ClassificationModel, self).__init__(opt)
        self.net = build_network(opt['network_g']).to(self.device)
        self.current_epoch = 0

        # Configurações de Two-Stage e Differential LRs
        train_opt = opt.get('train', {})
        self.two_stage_opt = train_opt.get('two_stage', {})
        self.two_stage_enabled = bool(self.two_stage_opt.get('enabled', False))
        self.freeze_epochs = int(self.two_stage_opt.get('freeze_epochs', 5))
        self.diff_lr_opt = train_opt.get('differential_lr', {})
        self.diff_lr_enabled = bool(self.diff_lr_opt.get('enabled', False))

        # Configurações de Mixup e CutMix
        reg_opt = train_opt.get('regularization', {})
        self.mixup_opt = reg_opt.get('mixup_cutmix', {})
        self.mixup_enabled = bool(self.mixup_opt.get('enabled', False))
        self.num_classes = int(opt['network_g'].get('num_classes', 15))

        if self.mixup_enabled:
            mixup_alpha = float(self.mixup_opt.get('mixup_alpha', 0.8))
            cutmix_alpha = float(self.mixup_opt.get('cutmix_alpha', 1.0))
            self.cutmix_or_mixup = v2.RandomChoice([
                v2.MixUp(alpha=mixup_alpha, num_classes=self.num_classes),
                v2.CutMix(alpha=cutmix_alpha, num_classes=self.num_classes),
            ])

        if self.is_train:
            self.init_training_settings()

    def _get_parameter_groups(self, stage: int = 1):
        """
        Separa os parâmetros em grupos com taxas de aprendizado diferenciais.
        Stage 1: Apenas a cabeça linear é treinada (backbone congelado).
        Stage 2: Backbone descongelado com taxas diferenciais.
        """
        weight_decay = float(self.opt['train'].get('weight_decay', 0.01))

        head_params = []
        shallow_params = []
        deep_params = []

        for name, param in self.net.named_parameters():
            if any(h in name for h in ['fc', 'classifier', 'head']):
                head_params.append(param)
            elif any(s in name for s in ['conv1', 'bn1', 'layer1']):
                shallow_params.append(param)
            else:
                deep_params.append(param)

        if stage == 1 and self.two_stage_enabled:
            # Congelar backbone
            for p in shallow_params + deep_params:
                p.requires_grad = False
            for p in head_params:
                p.requires_grad = True

            head_lr = float(self.two_stage_opt.get('head_lr', self.opt['train'].get('lr_g', 1e-3)))
            return [{'params': head_params, 'lr': head_lr, 'weight_decay': weight_decay}]

        # Stage 2 ou Differential LR direto
        for p in shallow_params + deep_params + head_params:
            p.requires_grad = True

        if self.diff_lr_enabled:
            s_lr = float(self.diff_lr_opt.get('backbone_shallow_lr', 1e-5))
            d_lr = float(self.diff_lr_opt.get('backbone_deep_lr', 5e-5))
            h_lr = float(self.diff_lr_opt.get('head_lr', 5e-4))
            return [
                {'params': shallow_params, 'lr': s_lr, 'weight_decay': weight_decay},
                {'params': deep_params, 'lr': d_lr, 'weight_decay': weight_decay},
                {'params': head_params, 'lr': h_lr, 'weight_decay': weight_decay},
            ]

        # Padrão plano
        base_lr = float(self.opt['train'].get('lr_g', 1e-4))
        return [{'params': self.net.parameters(), 'lr': base_lr, 'weight_decay': weight_decay}]

    def init_training_settings(self):
        self.net.train()
        train_opt = self.opt['train']
        label_smoothing = float(train_opt.get('label_smoothing', 0.1))
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(self.device)

        # Montar grupos de parâmetros
        stage = 1 if self.two_stage_enabled else 2
        param_groups = self._get_parameter_groups(stage=stage)
        self.optimizer_g = torch.optim.AdamW(param_groups)
        self.optimizers = [self.optimizer_g]

        # Configurar Schedulers
        sched_opt = train_opt.get('scheduler', {})
        sched_type = sched_opt.get('type', 'CosineAnnealingWarmup')
        total_epochs = int(train_opt.get('total_epochs', 35))

        if sched_type == 'CosineAnnealingWarmup':
            warmup_epochs = int(sched_opt.get('warmup_epochs', 3))
            min_lr = float(sched_opt.get('min_lr', 1e-6))
            cosine_epochs = max(1, total_epochs - warmup_epochs)
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer_g, start_factor=0.1, total_iters=warmup_epochs
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_g, T_max=cosine_epochs, eta_min=min_lr
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer_g, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
        else:
            step_size = int(train_opt.get('scheduler_step_size', 8))
            gamma = float(train_opt.get('scheduler_gamma', 0.2))
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer_g, step_size=step_size, gamma=gamma
            )

        self.schedulers = [self.scheduler]

    def _unfreeze_backbone_for_stage_2(self):
        """Transição do Estágio 1 (Head Warmup) para o Estágio 2 (Fine-tuning conjunto)."""
        print(f"\n[ClassificationModel] Transição de Estágio: Descongelando backbone na época {self.current_epoch}!")
        param_groups = self._get_parameter_groups(stage=2)
        weight_decay = float(self.opt['train'].get('weight_decay', 0.01))

        # Reconfigurar otimizador para os 3 grupos com differential LR
        self.optimizer_g = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        self.optimizers = [self.optimizer_g]

        # Reiniciar scheduler cossenoidal para as épocas restantes
        total_epochs = int(self.opt['train'].get('total_epochs', 35))
        remaining_epochs = max(1, total_epochs - self.current_epoch)
        min_lr = float(self.opt['train'].get('scheduler', {}).get('min_lr', 1e-6))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g, T_max=remaining_epochs, eta_min=min_lr
        )
        self.schedulers = [self.scheduler]

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        self.current_epoch += 1
        if self.two_stage_enabled and self.current_epoch == self.freeze_epochs:
            self._unfreeze_backbone_for_stage_2()
        else:
            for scheduler in self.schedulers:
                scheduler.step()

    def feed_data(self, data):
        self.x = data['x'].to(self.device)
        self.y = data['y'].to(self.device)

        # Aplicar Mixup/CutMix em treinamento
        if self.net.training and self.mixup_enabled:
            # v2.MixUp / CutMix aceita tensores no formato [B, C, H, W] e [B]
            self.x, self.y = self.cutmix_or_mixup(self.x, self.y.long())
        else:
            self.y = self.y.long()

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net(self.x)
        loss = self.criterion(self.output, self.y)
        loss.backward()

        # Gradient Clipping obrigatório para estabilidade numérica
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer_g.step()

        if self.y.ndim > 1:
            # Alvos soft com Mixup
            target_indices = torch.argmax(self.y, dim=1)
        else:
            target_indices = self.y

        predictions = torch.argmax(self.output, dim=1)
        accuracy = (predictions == target_indices).float().mean()

        self.log_dict = OrderedDict()
        self.log_dict['loss'] = loss.item()
        self.log_dict['acc'] = accuracy.item()

    def test(self):
        self.net.eval()
        with torch.no_grad():
            self.output = self.net(self.x)
        self.net.train()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['prediction'] = self.output.detach().cpu()
        out_dict['probabilities'] = torch.softmax(self.output, dim=1).detach().cpu()
        if self.y.ndim > 1:
            out_dict['target'] = torch.argmax(self.y, dim=1).detach().cpu()
        else:
            out_dict['target'] = self.y.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter, name=None):
        save_filename = f"{name or 'epoch'}_{epoch}.pth"
        models_dir = os.path.join(self.opt['path']['experiments_root'], 'models')
        os.makedirs(models_dir, exist_ok=True)
        save_path = os.path.join(models_dir, save_filename)
        torch.save(
            {
                'epoch': epoch,
                'current_iter': current_iter,
                'network': self.net.state_dict(),
                'optimizer': self.optimizer_g.state_dict(),
            },
            save_path,
        )

    def load(self):
        pass
