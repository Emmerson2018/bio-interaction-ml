import yaml
import os

def parse_options(opt_path, is_train=True):
    """Carrega o arquivo YAML de opções."""
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    opt['is_train'] = is_train

    # Exportar caminhos úteis
    root_path = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
    opt['path'] = {
        'root': root_path,
        'experiments_root': os.path.join(root_path, 'experiments', opt['name'])
    }
    
    # Criar diretórios de experimento
    os.makedirs(opt['path']['experiments_root'], exist_ok=True)
    os.makedirs(os.path.join(opt['path']['experiments_root'], 'models'), exist_ok=True)
    os.makedirs(os.path.join(opt['path']['experiments_root'], 'visualization'), exist_ok=True)

    return opt
