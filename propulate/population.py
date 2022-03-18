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

    def __eq__(self, ind):
        if self != ind or self.generation != ind.generation or self.rank != ind.rank or self.loss != ind.loss or self.active != ind.active or self.isle != ind.isle: #or self.migration_steps != ind.migration_steps:
            return False
        return True
