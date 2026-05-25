import os
from collections import OrderedDict

import torch
import torch.nn as nn

from base_tool.archs import build_network
from base_tool.models.base_model import BaseModel
from base_tool.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class ClassificationModel(BaseModel):
    def __init__(self, opt):
        super(ClassificationModel, self).__init__(opt)
        self.net = build_network(opt['network_g']).to(self.device)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net.train()
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        optim_params = [param for param in self.net.parameters() if param.requires_grad]
        self.optimizer_g = torch.optim.AdamW(
            optim_params,
            lr=self.opt['train']['lr_g'],
            weight_decay=self.opt['train'].get('weight_decay', 0.01),
        )
        self.optimizers.append(self.optimizer_g)
        self.schedulers.append(
            torch.optim.lr_scheduler.StepLR(
                self.optimizer_g,
                step_size=self.opt['train'].get('scheduler_step_size', 8),
                gamma=self.opt['train'].get('scheduler_gamma', 0.2),
            )
        )

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

    def save(self, epoch, current_iter):
        save_filename = f'epoch_{epoch}.pth'
        save_path = os.path.join(self.opt['path']['experiments_root'], 'models', save_filename)
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
