from abc import ABC, abstractmethod

class BaseMetric(ABC):
    """Interface para todas as métricas de avaliação."""
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, prediction, target):
        """Calcula a métrica.
        Args:
            prediction: Saída do modelo.
            target: Ground truth.
        Returns:
            float: Valor da métrica.
        """
        pass
