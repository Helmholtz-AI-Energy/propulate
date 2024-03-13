from propulate import Surrogate
from propulate.population import Individual


class LogSurrogate(Surrogate):
    """
    Surrogate model with just a few debug prints to see what's going on
    when debugging.
    """

    def __init__(self):
        super().__init__()
        self.synthetic_id: int = 0
        self.best_loss: float = 10000.0

        print("LogSurrogate initialized")

    def start_run(self, ind: Individual):
        self.synthetic_id = 0
        print(f"LogSurrogate - Start run called on individual with keys: {ind.keys()} and values: {ind.values()}")

    def update(self, loss: float):
        if loss < self.best_loss:
            self.best_loss = loss
        print(f"LogSurrogate - Updated on id {self.synthetic_id} with loss: {loss} and best loss: {self.best_loss}")

    def cancel(self, loss: float) -> bool:
        print(f"LogSurrogate - Cancel called on id {self.synthetic_id} with loss: {loss}")
        self.synthetic_id += 1
        return False

    def merge(self, data: float):
        print(f"LogSurrogate - Merge called with best loss: {self.best_loss} and data: {data}")
        if data < self.best_loss:
            self.best_loss = data

    def data(self) -> float:
        print(f"LogSurrogate - Data called with best loss: {self.best_loss}")
        return self.best_loss
