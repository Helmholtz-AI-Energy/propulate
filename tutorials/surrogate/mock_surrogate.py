from propulate import Surrogate
from propulate.population import Individual


class MockSurrogate(Surrogate):
    """
    This surrogate model does nothing.
    It exists for testing purposes and comparing performance decline
    due to the additional overhead of using a surrogate.

    Methods
    -------
    start_run()
        Does nothing.
    update()
        Does nothing.
    cancel()
        Always returns False.
    merge()
        Does nothing.
    data()
        Returns 0.0.

    Notes
    -----
    The ``MockSurrogate`` class implements all methods from the ``Surrogate`` class.

    See Also
    --------
    :class:`Surrogate` : The parent class.
    """

    def __init__(self) -> None:
        super().__init__()

    def start_run(self, ind: Individual) -> None:
        pass

    def update(self, loss: float) -> None:
        pass

    def cancel(self, loss: float) -> bool:
        return False

    def merge(self, data: float) -> None:
        pass

    def data(self) -> float:
        return 0.0
