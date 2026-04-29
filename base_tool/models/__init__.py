from base_tool.utils.registry import MODEL_REGISTRY
from base_tool.utils.misc import import_modules_from_directory
import os

# Importa automaticamente todos os modelos para registrá-los
model_folder = os.path.dirname(os.path.abspath(__file__))
import_modules_from_directory(model_folder, 'base_tool.models')

def build_model(opt):
    """Constrói um modelo a partir das opções."""
    model_type = opt['model_type']
    model = MODEL_REGISTRY.get(model_type)(opt)
    return model
