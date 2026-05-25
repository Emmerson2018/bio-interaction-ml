from abc import ABC, abstractmethod

class BaseMetric(ABC):
    """Interface para todas as metricas de avaliacao."""
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, prediction, target):
        """Calcula a metrica.
        Args:
            prediction: Saida do modelo.
            target: Ground truth.
        Returns:
            float: Valor da metrica.
        """
        pass
