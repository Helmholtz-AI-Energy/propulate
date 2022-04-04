# TODO multi objective optimization?
# TODO invalidate loss, when entry is modified so this does not have to be done by the propagator
# TODO switch to ordered dict
# TODO genealogy
# TODO have ordinal vs categorical inferred from list vs set

from decimal import Decimal

class Individual(dict):
    def __init__(self, generation=None, rank=None):
        super(Individual, self).__init__(list())
        self.generation = generation # Equals each worker's iteration for continuous population in propulate.
        self.rank = rank
        self.loss = None    # NOTE set to None instead of inf since there are no comparisons
        self.active = True
        self.isle = None    # isle of origin
        self.current = None
        self.migration_steps = None

    def __repr__(self):
        rep = {key : f"{Decimal(self[key]):.2E}" for key in self}
        Active = "active" if self.active else "deactivated"
        return f"[{rep}, loss {Decimal(self.loss):.2E}, I{self.isle}, W{self.rank}, G{self.generation}, w{self.current}, m{self.migration_steps}, {Active}]"

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
        return compare_traits and self.loss == other.loss and self.generation == other.generation \
                and self.rank == other.rank and self.isle == other.isle and self.active == other.active
