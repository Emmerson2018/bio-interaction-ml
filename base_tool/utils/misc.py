import importlib
import os
from base_tool.utils.scandir import scandir

def import_modules_from_directory(dir_path, package_prefix):
    """Importa todos os módulos python de um diretório para disparar registros."""
    module_files = scandir(dir_path, suffix='.py')
    for file_path in module_files:
        file_name = os.path.basename(file_path)
        if file_name.startswith('__'):
            continue
        
        module_name = file_name[:-3] # remove .py
        importlib.import_module(f'{package_prefix}.{module_name}')
