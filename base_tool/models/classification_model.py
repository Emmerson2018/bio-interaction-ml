import os
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

from base_tool.archs import build_network
from base_tool.models.base_model import BaseModel
from base_tool.utils.registry import MODEL_REGISTRY


def _build_optimizer(params, train_opt):
    optimizer_opt = dict(train_opt.get('optimizer') or {})
    optimizer_type = optimizer_opt.pop('type', 'AdamW')
    lr = optimizer_opt.pop('lr', train_opt.get('lr_g', 3e-4))
    weight_decay = optimizer_opt.pop('weight_decay', train_opt.get('weight_decay', 0.01))

    if optimizer_type == 'AdamW':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **optimizer_opt)
    if optimizer_type == 'Adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, **optimizer_opt)
    if optimizer_type == 'SGD':
        momentum = optimizer_opt.pop('momentum', 0.9)
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum, **optimizer_opt)
    raise ValueError(f'Unsupported optimizer type: {optimizer_type}')


def _build_scheduler(optimizer, train_opt):
    scheduler_opt = dict(train_opt.get('scheduler') or {})
    scheduler_type = scheduler_opt.pop('type', 'StepLR')
    if scheduler_type in {None, 'none', 'None'}:
        return None
    if scheduler_type == 'StepLR':
        step_size = scheduler_opt.pop('step_size', train_opt.get('scheduler_step_size', 8))
        gamma = scheduler_opt.pop('gamma', train_opt.get('scheduler_gamma', 0.2))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, **scheduler_opt)
    if scheduler_type == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_opt)
    raise ValueError(f'Unsupported scheduler type: {scheduler_type}')


@MODEL_REGISTRY.register()
class ClassificationModel(BaseModel):
    def __init__(self, opt):
        super(ClassificationModel, self).__init__(opt)
        self.net = build_network(opt['network_g']).to(self.device)
        self.dataset_metadata = {}
        self._load_pretrained_network()
        runtime_opt = opt.get('runtime', {}) or {}
        self.mixed_precision = bool(runtime_opt.get('mixed_precision', False)) and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.mixed_precision)

        if self.is_train:
            self.init_training_settings()

    def _resolve_path(self, path):
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return Path(self.opt.get('path', {}).get('root', Path.cwd())) / candidate

    def _load_pretrained_network(self):
        pretrained_path = self.opt.get('network_g', {}).get('pretrained_checkpoint')
        if not pretrained_path:
            return
        checkpoint = torch.load(self._resolve_path(pretrained_path), map_location=self.device)
        source_state = checkpoint.get('network', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        target_state = self.net.state_dict()
        compatible_state = {}
        skipped = []
        for key, value in source_state.items():
            if key in target_state and target_state[key].shape == value.shape:
                compatible_state[key] = value
            else:
                skipped.append(key)
        missing, unexpected = self.net.load_state_dict(compatible_state, strict=False)
        self.pretrained_load_report = {
            'path': str(pretrained_path),
            'loaded_keys': len(compatible_state),
            'skipped_keys': skipped,
            'missing_keys': list(missing),
            'unexpected_keys': list(unexpected),
        }

    def init_training_settings(self):
        self.net.train()
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer_g = _build_optimizer(self.net.parameters(), self.opt['train'])
        self.optimizers.append(self.optimizer_g)
        scheduler = _build_scheduler(self.optimizer_g, self.opt['train'])
        if scheduler is not None:
            self.schedulers.append(scheduler)

    def set_dataset_metadata(self, metadata):
        self.dataset_metadata = dict(metadata or {})

    def on_epoch_start(self, epoch):
        unfreeze_epoch = self.opt.get('train', {}).get('unfreeze_epoch')
        if unfreeze_epoch is not None and epoch >= int(unfreeze_epoch) and hasattr(self.net, 'unfreeze_backbone'):
            self.net.unfreeze_backbone()

    def feed_data(self, data):
        self.x = data['x'].to(self.device)
        self.y = data['y'].long().to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        with torch.amp.autocast('cuda', enabled=self.mixed_precision):
            self.output = self.net(self.x)
            loss = self.criterion(self.output, self.y)
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer_g.step()

        predictions = torch.argmax(self.output, dim=1)
        accuracy = (predictions == self.y).float().mean()
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
        out_dict['target'] = self.y.detach().cpu()
        return out_dict

    def _checkpoint_payload(self, epoch, current_iter, metric_results=None):
        idx_to_class = self.dataset_metadata.get('idx_to_class', {})
        class_to_idx = self.dataset_metadata.get('class_to_idx', {})
        model_metadata = self.net.get_metadata() if hasattr(self.net, 'get_metadata') else {}
        return {
            'epoch': epoch,
            'current_iter': current_iter,
            'network': self.net.state_dict(),
            'optimizer': self.optimizer_g.state_dict(),
            'metrics': dict(metric_results or {}),
            'class_to_idx': dict(class_to_idx),
            'idx_to_class': dict(idx_to_class),
            'preprocessing': self.dataset_metadata.get('preprocessing') or model_metadata.get('preprocessing'),
            'model_metadata': model_metadata,
            'dataset_metadata': dict(self.dataset_metadata),
            'training_metadata': {
                'name': self.opt.get('name'),
                'seed': self.opt.get('train', {}).get('seed'),
                'deterministic': self.opt.get('train', {}).get('deterministic', False),
                'runtime': dict(self.opt.get('runtime', {}) or {}),
                'device': str(self.device),
                'mixed_precision': self.mixed_precision,
                'peak_vram_mb': round(torch.cuda.max_memory_allocated(self.device) / 1024 / 1024, 3)
                if self.device.type == 'cuda' else 0.0,
            },
        }

    def save(self, epoch, current_iter, filename=None, metric_results=None):
        save_filename = filename or f'epoch_{epoch}.pth'
        save_path = os.path.join(self.opt['path']['experiments_root'], 'models', save_filename)
        torch.save(self._checkpoint_payload(epoch, current_iter, metric_results), save_path)

    def load(self):
        pass
