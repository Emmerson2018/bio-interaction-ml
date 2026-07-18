from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """Interface para todos os Datasets.
    
    Pode conter logicas comuns de leitura de arquivos ou validacao de metadados.
    """
    def __init__(self, opt):
        super(BaseDataset, self).__init__()
        self.opt = opt

    def __len__(self):
        return 0

    def __getitem__(self, index):
        return {}
