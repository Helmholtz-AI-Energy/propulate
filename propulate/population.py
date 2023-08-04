from decimal import Decimal


class Individual(dict):
    def __init__(
            self,
            generation: int = -1,
            rank: int = -1
    ) -> None:
        """
        Initialize individual with given paramters.

        Parameters
        ----------
        generation: int
                    current generation (-1 if unset)
        rank: int
              rank (-1 if unset)
        """
        super(Individual, self).__init__(list())
        self.generation = generation  # Equals each worker's iteration for continuous population in propulate.
        self.rank = rank
        self.loss = None  # NOTE set to None instead of inf since there are no comparisons
        self.active = True
        self.island = -1  # island of origin
        self.current = -1  # current responsible worker
        self.migration_steps = -1  # number of migration steps performed
        self.migration_history = None  # migration history
        self.evaltime = None  # evaluation time
        self.evalperiod = None  #

    def __repr__(self):
        rep = {
            key: (
                f"{Decimal(self[key]):.2E}" if type(self[key]) == float else self[key]
            )
            for key in self
        }
        is_active = "active" if self.active else "deactivated"
        if self.loss is None:
            loss_str = f"{self.loss}"
        else:
            loss_str = f"{Decimal(float(self.loss)):.2E}"
        return (
            f"[{rep}, loss " + loss_str + f", I{self.island}, W{self.rank}, "
            f"G{self.generation}, {self.evaltime}, w{self.current}, m{self.migration_steps}, {is_active}]"
        )

    def __eq__(self, other):
        # Check if object to compare to is of the same class.
        assert isinstance(other, self.__class__)
        # Check equivalence of actual traits, i.e., hyperparameter values.
        compare_traits = True
        for key in self.keys():
            if self[key] == other[key]:
                continue
            else:
                compare_traits = False
                break
        # Additionally check for equivalence of attributes (except for `self.migration_steps` and `self.current`).
        return (
            compare_traits
            and self.loss == other.loss
            and self.generation == other.generation
            and self.rank == other.rank
            and self.island == other.island
            and self.active == other.active
        )

    def equals(self, other):
        # Check if object to compare to is of the same class.
        assert isinstance(other, self.__class__)
        # Check equivalence of traits, i.e., hyperparameter values.
        compare_traits = True
        for key in self.keys():
            if self[key] == other[key]:
                continue
            else:
                compare_traits = False
                break
        return compare_traits and self.loss == other.loss
