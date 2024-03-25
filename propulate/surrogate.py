from typing import TypeVar, Generic
from abc import ABC, abstractmethod

from propulate.population import Individual

T = TypeVar("T")


class Surrogate(ABC, Generic[T]):
    """
    Abstract Surrogate Model for Propulator.

    A surrogate model is used in Propulator to detect poor performing
    individuals by periodically checking the yield of the loss function
    during its evaluation.
    If the surrogate model determines that the current run will not result
    in an improvement, Propulator will cancel the run.

    Implementation Checklist:

    1. Yield from ``loss_fn`` is called periodically without any randomness.
    2. Merge is commutative.
    """

    @abstractmethod
    def __init__(self) -> None:
        """
        Initialize a new surrogate model.

        Propulate passes down a surrogate factory, as defined by the user,
        to each Propulator instance. That means ``__init__`` can be overwritten
        to take additional arguments.
        """
        pass

    @abstractmethod
    def start_run(self, ind: Individual) -> None:
        """
        Signals that a new run is about to start.
        This is called before the first yield from the ``loss_fn``.
        It is assumed that the individual is freshly created.
        Keep in mind that there might be (private) keys
        that are not related to limits, like the surrogate key '_s';
        key names related to limits could be provided to
        the Surrogate constructor if necessary.

        Parameters
        ----------
        ind : propulate.Individual
            The individual to be evaluated.
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
            The final loss of the current run.
        """
        pass

    @abstractmethod
    def cancel(self, loss: float) -> bool:
        """
        Evaluate surrogate to check if the current run should be cancelled.
        This will be called after every yield from the ``loss_fn``.

        Parameters
        ----------
        loss : float
            The loss of the most recent step.

        Returns
        -------
        bool
           If the surrogate model determines that the current run
           will not result in a lower loss than previous runs.
        """
        return False

    @abstractmethod
    def merge(self, data: T) -> None:
        """
        Merges the results of another surrogate model into itself.
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
        Returns all relevant information about the surrogate model
        for merging with another surrogate

        It most likely only needs to return the most recent loss
        from ``update()``.

        Returns
        -------
        T
            All relevant information to convey the current state
            of the surrogate model.

        """
        pass
