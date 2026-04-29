from abc import ABC, abstractmethod
import torch

class BaseModel(ABC):
    """Interface abstrata para todos os modelos.
    
    Define os hooks obrigatórios para o ciclo de vida do treinamento.
    """
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
