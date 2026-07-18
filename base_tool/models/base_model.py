from abc import ABC, abstractmethod
import torch

class BaseModel(ABC):
    """Interface abstrata para todos os modelos.
    
    Define os hooks obrigatórios para o ciclo de vida do treinamento.
    """
    def __init__(self, opt):
        self.opt = opt
        runtime_opt = opt.get('runtime', {}) or {}
        require_cuda = bool(runtime_opt.get('require_cuda', False))
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError('runtime.require_cuda=true, but torch.cuda.is_available() is false.')
        if torch.cuda.is_available() and runtime_opt.get('device', 'cuda') == 'cuda':
            cuda_device = int(runtime_opt.get('cuda_device', 0))
            if cuda_device >= torch.cuda.device_count():
                raise RuntimeError(f'Requested cuda:{cuda_device}, but only {torch.cuda.device_count()} CUDA device(s) are available.')
            self.device = torch.device(f'cuda:{cuda_device}')
            torch.cuda.set_device(self.device)
        else:
            if require_cuda:
                raise RuntimeError('CUDA was required, but runtime.device is not cuda.')
            self.device = torch.device('cpu')
        self.is_train = opt.get('is_train', True)
        self.schedulers = []
        self.optimizers = []

    @abstractmethod
    def feed_data(self, data):
        """Prepara os dados para a rede."""
        pass

    @abstractmethod
    def optimize_parameters(self, current_iter):
        """Passo de otimização (forward, loss, backward, step)."""
        pass

    @abstractmethod
    def test(self):
        """Inferência."""
        pass

    @abstractmethod
    def save(self, epoch, current_iter):
        """Salva checkpoints."""
        pass

    @abstractmethod
    def load(self):
        """Carrega checkpoints."""
        pass

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Atualiza a taxa de aprendizado usando os schedulers registrados."""
        for scheduler in self.schedulers:
            scheduler.step()

    def get_current_learning_rate(self):
        return [optimizer.param_groups[0]['lr'] for optimizer in self.optimizers]

    def get_current_losses(self):
        return getattr(self, 'log_dict', {})
