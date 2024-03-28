import numpy as np

from propulate import Surrogate
from propulate.population import Individual


class StaticSurrogate(Surrogate):
    """
    Surrogate model using the best known run as baseline.

    After the first run, each subsequent loss is compared to the baseline. Every run with a loss outside the margin of
    the baseline is cancelled.

    This model creates an internal index for the yielded losses during each run. That means the yield order and
    frequency has to be consistent between runs. Otherwise, the indices of the baseline run won't match.

    Loosely based on the paper:
    Use of Static Surrogates in Hyperparameter Optimization
    https://link.springer.com/article/10.1007/s43069-022-00128-w

    Attributes
    ----------
    synthetic_id : int
        An internal running index to keep track of the current runs loss index.
    margin : float
        A margin to be multiplied with the new incoming losses to compare them to the baseline.
    first_run : bool
        Flag for the first complete run.
    baseline : np.ndarray
        Chronological list of the losses of the best known run.
    current_run : np.ndarray
        Array to store the losses of the current run.

    Methods
    -------
    __init__()
        Overwrites the parents class's constructor to include the error margin.
    start_run()
        Reset the internal index and the current run array.
    update()
        Replace the baseline with the current run if the final loss is better.
    cancel()
        Cancel the current run if the loss is outside the margin of the baseline.
    merge()
        Merge an incoming baseline.
    data()
        Return the loss series of the best known run so far.

    Notes
    -----
    The ``StaticSurrogate`` class implements all methods from the ``Surrogate`` class.

    See Also
    --------
    :class:`Surrogate` : The parent class.
    """

    def __init__(self, margin: float = 0.8) -> None:
        """
        Initialize a static surrogate with a synthetic id, a margin, and empty arrays for baseline and current run.

        Parameters
        ----------
        margin : float, optional
            A margin on top of incoming losses for comparison with the baseline.
            The default is 0.8.
        """
        super().__init__()

        self.synthetic_id: int = 0
        self.margin: float = margin

        # Canceling with ``cancel`` is only allowed after the first complete run.
        # ``first_run`` keeps track of that.
        self.first_run: bool = True

        # Baseline is the best known run.
        self.baseline: np.ndarray = np.zeros((0), dtype=float)
        self.current_run: np.ndarray = np.zeros((0), dtype=float)

    def start_run(self, ind: Individual) -> None:
        """
        Reset the internal index and the current run array.

        Parameters
        ----------
        ind : Individual
            The individual containing the current configuration.
        """
        # Reset to new run.
        self.synthetic_id = 0
        # Reset current run with correct size.
        self.current_run = np.zeros((self.baseline.size), dtype=float)

    def update(self, loss: float) -> None:
        """
        Replace the baseline with the current run if the final loss is better, or if there is no prior run.

        Parameters
        ----------
        loss : float
            The (unused) final loss of the current run.
        """
        if self.first_run:
            self.baseline = self.current_run.copy()
            self.first_run = False
            return

        # Check if current run is better than baseline.
        if self.baseline[-1] > self.current_run[-1]:
            self.baseline = self.current_run.copy()

    def cancel(self, loss: float) -> bool:
        """
        Cancel the current run if the loss is outside the margin of the baseline.

        Parameters
        ----------
        loss : float
            The next interim loss of the current run.

        Returns
        -------
        bool
            True if the current run is cancelled, False otherwise.
        """
        self.synthetic_id += 1

        # Cancel is only allowed after the first complete run.
        if self.first_run:
            self.current_run = np.append(self.current_run, loss)
            return False

        # Append loss to current run.
        self.current_run[self.synthetic_id - 1] = loss

        # Cancel if current run is outside margin of baseline.
        if self.baseline[self.synthetic_id - 1] < loss * self.margin:
            return True

        return False

    def merge(self, data: np.ndarray) -> None:
        """
        Replace the baseline with the incoming run if the final loss is better, or if there is no prior run.

        Parameters
        ----------
        data : np.ndarray
            The loss series of the incoming run.
        """
        # No prior data to merge with
        if self.first_run:
            self.baseline = data.copy()
            self.first_run = False
            return

        # Merged run's final loss is better than baseline.
        if self.baseline[-1] < data[-1]:
            self.baseline = data.copy()

    def data(self) -> np.ndarray:
        """
        Return the loss series of the best known run so far.

        Returns
        -------
        np.ndarray
            The loss series of the best known run so far.
        """
        # Return best run so far.
        return self.baseline
