import torch
import torch.nn as nn
import os
from collections import OrderedDict
from base_tool.models.base_model import BaseModel
from base_tool.utils.registry import MODEL_REGISTRY
from base_tool.archs import build_network

@MODEL_REGISTRY.register()
class SimpleModel(BaseModel):
    def __init__(self, opt):
        super(SimpleModel, self).__init__(opt)
        
        # 1. Build Network
        self.net = build_network(opt['network_g'])
        self.net = self.net.to(self.device)
        
        # 2. Setup training
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net.train()
        
        # Loss
        self.cri_pix = nn.MSELoss().to(self.device)
        
        # Optimizer
        optim_params = []
        for v in self.net.parameters():
            if v.requires_grad:
                optim_params.append(v)
        
        self.optimizer_g = torch.optim.Adam(optim_params, lr=self.opt['train']['lr_g'])
        self.optimizers.append(self.optimizer_g)
        
        # Schedulers (Simplificado para POC)
        self.schedulers.append(torch.optim.lr_scheduler.StepLR(self.optimizer_g, step_size=10, gamma=0.1))

    def feed_data(self, data):
        self.x = data['x'].to(self.device)
        self.y = data['y'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net(self.x)
        
        loss = self.cri_pix(self.output, self.y)
        loss.backward()
        self.optimizer_g.step()
        
        self.log_dict = OrderedDict()
        self.log_dict['loss'] = loss.item()

    def test(self):
        self.net.eval()
        with torch.no_grad():
            self.output = self.net(self.x)
        self.net.train()

    def get_current_visuals(self):
        """Retorna predição e target para cálculo de métricas."""
        out_dict = OrderedDict()
        out_dict['prediction'] = self.output.detach().cpu()
        out_dict['target'] = self.y.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        save_filename = f'epoch_{epoch}.pth'
        save_path = os.path.join(self.opt['path']['experiments_root'], 'models', save_filename)
        torch.save(self.net.state_dict(), save_path)

    def load(self):
        pass # Implementar futuramente
