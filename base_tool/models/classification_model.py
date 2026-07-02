import os
from collections import OrderedDict

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

        if self.is_train:
            self.init_training_settings()

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
        self.output = self.net(self.x)
        loss = self.criterion(self.output, self.y)
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
            },
        }

    def save(self, epoch, current_iter, filename=None, metric_results=None):
        save_filename = filename or f'epoch_{epoch}.pth'
        save_path = os.path.join(self.opt['path']['experiments_root'], 'models', save_filename)
        torch.save(self._checkpoint_payload(epoch, current_iter, metric_results), save_path)

    def load(self):
        pass
