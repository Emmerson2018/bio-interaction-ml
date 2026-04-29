from base_tool.utils.registry import ARCH_REGISTRY
from base_tool.utils.misc import import_modules_from_directory
import os

# Importa automaticamente todas as arquiteturas
arch_folder = os.path.dirname(os.path.abspath(__file__))
import_modules_from_directory(arch_folder, 'base_tool.archs')

def build_network(opt):
    """Constrói uma rede neural (nn.Module)."""
    arch_type = opt['type']
    # Remove 'type' para passar os demais parâmetros como kwargs para o construtor da classe
    kwargs = {k: v for k, v in opt.items() if k != 'type'}
    network = ARCH_REGISTRY.get(arch_type)(**kwargs)
    return network
