import matplotlib.pyplot as plt
import os
from base_tool.visualization.base_visualizer import BaseVisualizer
from base_tool.utils.registry import VISUALIZATION_REGISTRY

@VISUALIZATION_REGISTRY.register()
class LossVisualizer(BaseVisualizer):
    """Visualizador genérico de métricas escalares (Losses, Acc, etc)."""
    def __init__(self, opt):
        super(LossVisualizer, self).__init__(opt)
        self.histories = {} # Dicionário de listas para múltiplas curvas

    def visualize(self, current_iter, log_dict):
        # log_dict pode conter {'loss_G': 0.1, 'loss_D': 0.2, 'acc': 0.9}
        for key, value in log_dict.items():
            if key not in self.histories:
                self.histories[key] = []
            self.histories[key].append((current_iter, value))
        
        # Gera um plot com múltiplas curvas
        plt.figure(figsize=(10, 5))
        for key, history in self.histories.items():
            iters, values = zip(*history)
            plt.plot(iters, values, label=key)
        
        plt.xlabel('Iterations')
        plt.ylabel('Values')
        plt.title(f'Training Metrics - {self.opt["name"]}')
        plt.legend()
        plt.grid(True)
        
        save_file = os.path.join(self.save_path, 'training_metrics.png')
        plt.savefig(save_file)
        plt.close()
