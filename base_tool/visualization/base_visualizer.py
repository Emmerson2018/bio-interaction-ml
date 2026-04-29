from abc import ABC, abstractmethod
import os

class BaseVisualizer(ABC):
    """Interface para todos os visualizadores da toolbox."""
    def __init__(self, opt):
        self.opt = opt
        self.save_path = os.path.join(opt['path']['experiments_root'], 'visualization')
        os.makedirs(self.save_path, exist_ok=True)

    @abstractmethod
    def visualize(self, current_iter, log_dict):
        """Método principal. 
        log_dict contém tudo: losses, metrics, lrs, etc.
        """
        pass
