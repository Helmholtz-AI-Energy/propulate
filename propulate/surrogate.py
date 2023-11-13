class Surrogate:
    """
    Interface Surrogate model for Propulator.
    """

    def __init__(self):
        """
        """
        pass

    def update(self, loss: float):
        """
        Update the surrogate model with the given loss (or other data?)
        """
        return NotImplementedError

    def cancel(self, loss: float) -> bool:
        """
        Evaluate Surrogate.

        Should the current training run, given the next loss, be cancelled?
        """
        return False

    def merge(self, new: 'Surrogate'):
        """
        Given 2 suurrogate models, merge them into one.

        Implementation of merge has to be commutative!
        """
        return NotImplementedError

    def data(self) -> dict:
        """
        Data returns all relevant information about the surrogate model
        so it can be used to merge with another surrogate
        """
        return {}
