# from collections import UserDict
from decimal import Decimal
from typing import Union

import numpy as np


class Individual:
    """An individual represents a candidate solution to the considered optimization problem."""

    def __init__(
        self,
        position: Union[dict, np.ndarray],
        limits: dict,
        velocity: np.ndarray = None,
        generation: int = -1,
        rank: int = -1,
    ) -> None:
        """
        Initialize an individual with given parameters.

        Parameters
        ----------
        generation : int
            The current generation (-1 if unset).
        rank : int
            The rank (-1 if unset).
        """
        # TODO compare keys of position to keys of limits
        self.limits = limits
        for key in limits:
            if key.startswith("_"):
                raise ValueError("Keys starting with '_' are reserved.")
        self.types = {key: type(limits[key][0]) for key in limits}
        offset = 0
        self.offsets = {}
        for key in limits:
            self.offsets[key] = offset
            if isinstance(limits[key][0], str):
                offset += len(limits[key])
            else:
                offset += 1

        if isinstance(position, np.ndarray):
            self.position = position
            if len(position) != offset:
                raise ValueError(
                    "Individual position not compatible with given search space limits."
                )
            # super(Individual, self).__init__({k: self[k] for k in self.limits})
            self.mapping = {k: self[k] for k in self.limits}
        else:
            # super(Individual, self).__init__(position)
            # super(Individual, self).__init__()
            self.mapping = position
            self.position = np.zeros(offset)
            for key in position:
                self[key] = position[key]

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

        self.velocity = velocity
        if self.velocity is not None:
            if not self.position.shape == self.velocity.shape:
                raise ValueError("Position and velocity shape do not match.")

    def __getitem__(self, key):
        """Return decoded value for input key."""
        # super(Individual, self).__getitem__(key)
        if key.startswith("_"):
            # return super(Individual, self).__getitem__(key)
            return self.mapping[key]
        else:
            if self.types[key] == float:
                return self.position[self.offsets[key]].item()
            elif self.types[key] == int:
                return np.rint(self.position[self.offsets[key]]).item()
            elif self.types[key] == str:
                offset = self.offsets[key]
                upper = self.offsets[key] + len(self.limits[key])
                return self.limits[key][np.argmax(self.position[offset:upper]).item()]
            else:
                raise ValueError("Unknown type")

    def __setitem__(self, key, newvalue):
        """Encode and set value for given key."""
        # super(Individual, self).__setitem__(key, newvalue)
        self.mapping[key] = newvalue
        if key.startswith("_"):
            pass
        else:
            if key not in self.limits:
                raise ValueError("Unknown gene.")
            if self.types[key] == float:
                assert isinstance(newvalue, float)
                self.position[self.offsets[key]] = newvalue
            elif self.types[key] == int:
                assert isinstance(newvalue, int)
                self.position[self.offsets[key]] = float(newvalue)
            elif self.types[key] == str:
                assert newvalue in self.limits[key]
                offset = self.offsets[key]
                upper = len(self.limits[key])
                self.position[offset:upper] = np.array([0])
                self.position[offset + self.limits[key].index(newvalue)] = 1.0
            else:
                raise ValueError("Unknown type")

    def __delitem__(self, key):
        """Do not implement deleting items."""
        if key in self.limits:
            raise ValueError()
        self.mapping.__delitem__(key)

    def __len__(self):
        """Give number of genes i.e. the dimension of the parameter space. Each categorical variable adds only one dimension."""
        return len(self.limits)

    def values(self):
        """Return dict values view."""
        return self.mapping.values()

    def items(self):
        """Return dict items view."""
        return self.mapping.items()

    def keys(self):
        """Return dict keys view."""
        return self.mapping.keys()

    def __repr__(self) -> str:
        """Return string representation of an ``Individual`` instance."""
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

    def __iter__(self):
        """Return standard iterator."""
        for key in self.limits:
            yield key

    def __eq__(self, other) -> bool:
        """
        Define equality operator ``==`` for class ``Individual``.

        Checks for equality of traits, loss, generation, worker rank, birth island, and active status. Other attributes,
        like migration steps, are not considered.

        Parameters
        ----------
        other : Individual
            Other individual to compare individual under consideration to.

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
            other individual to compare individual under consideration to

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
