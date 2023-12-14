from propulate import Surrogate
import numpy as np


class StaticSurrogate(Surrogate):
    """
    Surrogate model using the best known run as baseline.
    After the first run, each subsequent onw is compared to the baseline.
    Every run outside a margin of the baseline is cancelled.

    This model assumes regular yields between training runs,
    otherwise the indices of the baseline run won't match.
    """

    def __init__(self, margin: float = 0.8):
        super().__init__()

        self.synthetic_id: int = 0
        self.margin: float = margin

        # cancel is only allowed after the first complete run
        # first_run keeps track of that
        self.first_run: bool = True

        # baseline is the best known run
        self.baseline: np.ndarray = np.zeros((0), dtype=float)
        self.current_run: np.ndarray = np.zeros((0), dtype=float)

    def update(self, loss: float):
        if self.first_run:
            self.baseline = self.current_run.copy()
            self.first_run = False

        # check if current run is better than baseline
        # current run is initialized with zeros, not finished runs are ignored
        if self.baseline[-1] < self.current_run[-1]:
            self.baseline = self.current_run.copy()

        # reset to new run
        self.synthetic_id = 0

        self.current_run = np.zeros((self.baseline.size), dtype=float)

    def cancel(self, loss: float) -> bool:
        self.synthetic_id += 1

        # cancel is only allowed after the first complete run
        if self.first_run:
            return False

        # cancel if current run is outside margin of baseline
        if self.baseline[self.synthetic_id - 1] * self.margin > loss:
            return True

        self.current_run[self.synthetic_id - 1] = loss
        return False

    def merge(self, data: np.ndarray):
        if self.first_run:
            self.baseline = data.copy()
            self.first_run = False
            return

        # merged runs final loss is better than baseline
        if self.baseline[-1][1] < data[-1][1]:
            self.baseline = data.copy()

    def data(self) -> np.ndarray:
        # return best run so far
        return self.baseline
