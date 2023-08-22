from decimal import Decimal


class Individual(dict):
    """
    An individual represents a candidate solution to the considered optimization problem.
    """

    def __init__(self, generation: int = -1, rank: int = -1) -> None:
        """
        Initialize an individual with given parameters.

        Parameters
        ----------
        generation: int
                    current generation (-1 if unset)
        rank: int
              rank (-1 if unset)
        """
        super(Individual, self).__init__(list())
        self.generation = generation  # Equals each worker's iteration for continuous population in Propulate.
        self.rank = rank
        self.loss = None  # Set to None instead of inf since there are no comparisons
        self.active = True
        self.island = -1  # island of origin
        self.current = -1  # current responsible worker
        self.migration_steps = -1  # number of migration steps performed
        self.migration_history = None  # migration history
        self.evaltime = None  # evaluation time
        self.evalperiod = None  # evaluation duration

    def __repr__(self) -> str:
        """
        String representation of an ``Individual`` instance.
        """
        rep = {
            key: (
                f"{Decimal(self[key]):.2E}"
                if isinstance(self[key], float)
                else self[key]
            )
            for key in self
        }
        if self.loss is None:
            loss_str = f"{self.loss}"
        else:
            loss_str = f"{Decimal(float(self.loss)):.2E}"
        return (
            f"[{rep}, loss "
            + loss_str
            + f", island {self.island}, worker {self.rank}, "
            f"generation {self.generation}]"
        )

    def __eq__(self, other) -> bool:
        """
        Define equality operator ``==`` for class ``Individual``.

        Checks for equality of traits, loss, generation, worker rank, birth island, and active status. Other attributes,
        like migration steps, are not considered.

        Parameters
        ----------
        other: Individual
               Other individual to compare individual under consideration to

        Returns
        -------
        bool
            True if individuals are the same, false if not.

        Raises
        ------
        TypeError
            If other is not an instance or subclass of ``Individual``.
        """
        # Check if object to compare to is of the same class.
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"{other} not an instance of `Individual` but {type(other)}."
            )

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

    def equals(self, other) -> bool:
        """
        Define alternative equality check for class ``Individual``.

        Checks for equality of traits and loss. Other attributes, like birth island or generation, are not considered.

        Parameters
        ----------
        other: Individual
               Other individual to compare individual under consideration to

        Returns
        -------
        bool
            True if individuals are the same, false if not.

        Raises
        ------
        TypeError
            If other is not an instance or subclass of ``Individual``.
        """
        # Check if object to compare to is of the same class.
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"{other} not an instance of `Individual` but {type(other)}."
            )
        # Check equivalence of traits, i.e., hyperparameter values.
        compare_traits = True
        for key in self.keys():
            if self[key] == other[key]:
                continue
            else:
                compare_traits = False
                break
        return compare_traits and self.loss == other.loss
