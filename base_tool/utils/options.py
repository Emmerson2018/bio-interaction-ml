import yaml
import os

def parse_options(opt_path, is_train=True):
    """Carrega o arquivo YAML de opcoes."""
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    opt['is_train'] = is_train

    # Exportar caminhos uteis
    root_path = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
    exp_root = os.path.join(root_path, 'experiments', opt['name'])
    opt['path'] = {
        'root': root_path,
        'experiments_root': exp_root,
        'models': os.path.join(exp_root, 'models'),
        'visualization': os.path.join(exp_root, 'visualization'),
    }
    
    # Criar diretorios de experimento
    os.makedirs(opt['path']['experiments_root'], exist_ok=True)
    os.makedirs(opt['path']['models'], exist_ok=True)
    os.makedirs(opt['path']['visualization'], exist_ok=True)

    return opt
