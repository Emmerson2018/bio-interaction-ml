from base_tool.utils.registry import METRIC_REGISTRY
from base_tool.utils.misc import import_modules_from_directory
import os

# Auto-import de métricas
metric_folder = os.path.dirname(os.path.abspath(__file__))
import_modules_from_directory(metric_folder, 'base_tool.metrics')

def build_metric(opt):
    """Constrói uma métrica."""
    metric_type = opt['type']
    # Passa o resto dos parâmetros do YAML para o construtor se houver
    kwargs = {k: v for k, v in opt.items() if k != 'type'}
    return METRIC_REGISTRY.get(metric_type)(**kwargs)
