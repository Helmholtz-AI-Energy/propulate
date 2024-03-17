from propulate import Surrogate
from propulate.population import Individual


class MockSurrogate(Surrogate):
    """
    This surrogate model does nothing.
    It exists for testing purposes and comparing performance decline
    due to the additional overhead of using a surrogate.
    """

    def __init__(self):
        super().__init__()

    def start_run(self, ind: Individual):
        pass

    def update(self, loss: float):
        pass

    def cancel(self, loss: float) -> bool:
        return False

    def merge(self, data: float):
        pass

    def data(self) -> float:
        return 0.0
