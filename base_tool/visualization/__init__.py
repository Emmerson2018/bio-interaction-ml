from base_tool.utils.registry import VISUALIZATION_REGISTRY
from base_tool.utils.misc import import_modules_from_directory
import os

# Importa automaticamente todos os visualizadores
viz_folder = os.path.dirname(os.path.abspath(__file__))
import_modules_from_directory(viz_folder, 'base_tool.visualization')

def build_visualizer(opt):
    """Constrói um visualizador."""
    viz_type = opt['type']
    visualizer = VISUALIZATION_REGISTRY.get(viz_type)(opt)
    return visualizer
