# TODO multi objective optimization?
# TODO invalidate loss, when entry is modified so this does not have to be done by the propagator
# TODO switch to ordered dict
# TODO genealogy
# TODO have ordinal vs categorical inferred from list vs set

class Individual(dict):
    def __init__(self, generation=None, rank=None, traits=[]):
        super(Individual, self).__init__(list())
        self.generation = generation # For continuous population in propulate, generation just equals iteration for each individual.
        self.rank = rank
        self.loss = None  # NOTE set to None instead of inf since there are no comparisons
        self.active = True
        self.isle = None # birth island of origin

    def __repr__(self):
        return super().__repr__()+f", isle {self.isle}, worker {self.rank}, generation {self.generation}, active {self.active}"
