import numpy as np
from propulate import Surrogate
from propulate.population import Individual


class StaticSurrogate(Surrogate):
    """
    Surrogate model using the best known run as baseline.
    After the first run, each subsequent loss is compared to the baseline.
    Every run with a loss outside the margin of the baseline is cancelled.

    This model creates an internal index for the yielded losses during each run.
    That means the yield order and frquency has to be consistent between runs.
    Otherwise, the indices of the baseline run won't match.

    Loosely based on the paper:
    Use of Static Surrogates in Hyperparameter Optimization
    https://link.springer.com/article/10.1007/s43069-022-00128-w
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

    def start_run(self, ind: Individual):
        # reset to new run
        self.synthetic_id = 0
        # reset current run with correct size
        self.current_run = np.zeros((self.baseline.size), dtype=float)

    def update(self, loss: float):
        if self.first_run:
            self.baseline = self.current_run.copy()
            self.first_run = False
            return

        # check if current run is better than baseline
        if self.baseline[-1] > self.current_run[-1]:
            self.baseline = self.current_run.copy()

    def cancel(self, loss: float) -> bool:
        self.synthetic_id += 1

        # cancel is only allowed after the first complete run
        if self.first_run:
            self.current_run = np.append(self.current_run, loss)
            return False

        # append loss to current run
        self.current_run[self.synthetic_id - 1] = loss

        # cancel if current run is outside margin of baseline
        if self.baseline[self.synthetic_id - 1] < loss * self.margin:
            return True

        return False

    def merge(self, data: np.ndarray):
        # no prior data to merge with
        if self.first_run:
            self.baseline = data.copy()
            self.first_run = False
            return

        # merged runs final loss is better than baseline
        if self.baseline[-1] < data[-1]:
            self.baseline = data.copy()

    def data(self) -> np.ndarray:
        # return best run so far
        return self.baseline
