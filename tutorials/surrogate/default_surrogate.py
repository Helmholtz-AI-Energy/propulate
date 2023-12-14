from propulate import Surrogate


class DefaultSurrogate(Surrogate):
    """
    This surrogate model does nothing.
    It exists for testing purposes and comparing performance decline
    due to the additional overhead of using a surrogate.
    """

    def __init__(self):
        super().__init__()

    def update(self, loss: float):
        pass

    def cancel(self, loss: float) -> bool:
        return False

    def merge(self, data: float):
        pass

    def data(self) -> float:
        return 0.0
