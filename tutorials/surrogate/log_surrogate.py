from propulate import Surrogate
from propulate.population import Individual


class LogSurrogate(Surrogate):
    """
    Surrogate model with just a few debug prints to see what's going on when debugging.

    Prints once for every method call. Additionally, it keeps track of the best final loss so far and an id that
    increments for every cancel call. This makes it easier to read the debug prints.

    Attributes
    ----------
    synthetic_id : int
        The id of the synthetic run.
    best_loss : float
        The best loss so far.

    Methods
    -------
    start_run()
        Resets the id and prints the keys and values of the individual.
    update()
        Prints the loss and the best loss so far. Updates the best loss if necessary.
    cancel()
        Increments the id and prints the loss. Always returns False.
    merge()
        Prints the best loss and the data. Updates the best loss if necessary.
    data()
        Prints the best loss and returns it.

    Notes
    -----
    The ``LogSurrogate`` class implements all methods from the ``Surrogate`` class.

    See Also
    --------
    :class:`Surrogate` : The parent class.
    """

    def __init__(self) -> None:
        """Initialize the ``LogSurrogate`` with a synthetic id and a high best loss."""
        super().__init__()
        self.synthetic_id: int = 0
        self.best_loss: float = 10000.0

        print("LogSurrogate initialized")

    def start_run(self, ind: Individual) -> None:
        """
        Reset the synthetic id and print the keys and values of the individual.

        Parameters
        ----------
        ind : Individual
            The individual containing the current configuration.
        """
        self.synthetic_id = 0
        print(
            f"LogSurrogate - Start run called on individual with keys: {ind.keys()} and values: {ind.values()}"
        )

    def update(self, loss: float) -> None:
        """
        Print the loss and the best loss so far. Update the best loss if necessary.

        Parameters
        ----------
        loss : float
            The final loss of the current run.
        """
        if loss < self.best_loss:
            self.best_loss = loss
        print(
            f"LogSurrogate - Updated on id {self.synthetic_id} with loss: {loss} and best loss: {self.best_loss}"
        )

    def cancel(self, loss: float) -> bool:
        """
        Increment the synthetic id and print the loss. Always return False.

        Parameters
        ----------
        loss : float
            The next interim loss of the current run.

        Returns
        -------
        bool
            Always False.
        """
        print(
            f"LogSurrogate - Cancel called on id {self.synthetic_id} with loss: {loss}"
        )
        self.synthetic_id += 1
        return False

    def merge(self, data: float) -> None:
        """
        Print the best loss and the data. Update the best loss if necessary.

        Parameters
        ----------
        data : float
            The best loss of the merged run.
        """
        print(
            f"LogSurrogate - Merge called with best loss: {self.best_loss} and data: {data}"
        )
        if data < self.best_loss:
            self.best_loss = data

    def data(self) -> float:
        """
        Print the best loss and return it.

        Returns
        -------
        float
            The best loss so far.
        """
        print(f"LogSurrogate - Data called with best loss: {self.best_loss}")
        return self.best_loss
