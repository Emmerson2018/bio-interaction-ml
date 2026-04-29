class Registry():
    """Sistema de registro para mapear strings para classes.
    
    Inspirado no padrão utilizado em frameworks como BasicSR e MMCV.
    """
    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj):
        if name in self._obj_map:
            raise KeyError(f'{name} já está registrado em {self._name}')
        self._obj_map[name] = obj

    def register(self, obj=None):
        """Registra um objeto. Pode ser usado como decorador."""
        if obj is not None:
            self._do_register(obj.__name__, obj)
            return obj

        def _register(cls):
            self._do_register(cls.__name__, cls)
            return cls

        return _register

    def get(self, name):
        """Recupera um objeto pelo nome."""
        if name not in self._obj_map:
            raise KeyError(f'{name} não encontrado em {self._name}')
        return self._obj_map[name]

    def __contains__(self, name):
        return name in self._obj_map

    def __repr__(self):
        return f'Registry(name={self._name}, items={list(self._obj_map.keys())})'

# Definição dos registros globais
ARCH_REGISTRY = Registry('ARCH')
MODEL_REGISTRY = Registry('MODEL')
DATASET_REGISTRY = Registry('DATASET')
LOSS_REGISTRY = Registry('LOSS')
METRIC_REGISTRY = Registry('METRIC')
VISUALIZATION_REGISTRY = Registry('VISUALIZATION')
