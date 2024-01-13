from typing import TypeVar, Generic
from abc import ABC, abstractmethod

from propulate.population import Individual

T = TypeVar("T")


class Surrogate(ABC, Generic[T]):
    """
    Abstract Surrogate model for Propulator.

    A surrogate model is used to detect poor performing individuals, by
    periodically checking the yield of the loss function during its evaluation.
    If the surrogate model determines that the current run will not result
    in an improvement, Propulator will cancel the run.

    Implementation Checklist:
    * Yield from loss_fn is called periodically without any randomness.
    * Merge has to be commutative!
    """

    @abstractmethod
    def __init__(self) -> None:
        """
        Initialize a new surrogate model.
        """
        pass

    @abstractmethod
    def start_run(
        self,
        ind: Individual
    ) -> None:
        """
        Signalizes that a new run is about to start.
        This is called before the first yield from the loss_fn.
        Assume that the individual is freshly created.
        Keep in mind that there might be (private) keys
        that are not related to limits, like the surrogate_key '_s'.
        Key names (limits) could be given in Surrogate constructor if needed.

        Parameters
        ----------
        ind: Individual
             The to be evaluated individual
        """
        pass

    @abstractmethod
    def update(self, loss: float) -> None:
        """
        Update the surrogate model with the final loss.
        Indicative that the current run has finished.

        Parameters
        ----------
        loss : float
               final loss of the current run
        """
        pass

    @abstractmethod
    def cancel(self, loss: float) -> bool:
        """
        Evaluate Surrogate to check if the current run should be cancelled.
        This will be called after every yield from the loss_fn.

        Parameters
        ----------
        loss : float
               loss of the most recent step

        Returns
        -------
        bool
            if the surrogate model determines that the current run
            will not result in a lower loss than previous runs
        """
        return False

    @abstractmethod
    def merge(self, data: T) -> None:
        """
        Merges the results of another Surrogate model into itself.
        Used to synchronize surrogate models from different Propulators.

        Implementation of merge has to be commutative!
        Otherwise the different surrogate models will diverge.

        Parameters
        ----------
        data : T
               All relevant information to update its model
               to the same state as the origin of the data.
        """
        pass

    @abstractmethod
    def data(self) -> T:
        """
        Data returns all relevant information about the surrogate model
        so it can be used to merge with another surrogate.

        It most likely only needs to return the most recent loss from update()

        Returns
        -------
        T
            All relevent information to convey the current state
            of the surrogate model.

        """
        pass
