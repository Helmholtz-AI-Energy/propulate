import random
import copy

from .population import Individual

# TODO add simplex step?

def _check_compatible(out1, in2):
    """
    Check compability of two propagators for stacking them together sequentially with Cascade().
    """
    return out1 == in2 or in2==-1


class Propagator():
    """
    Abstract base class for all propagators, i.e., evolutionary operators, in Propulate.

    Take a collection of individuals and use them to breed a new collection of individuals.
    """
    def __init__(self, parents=0, offspring=0):
        """
        Constructor of Propagator class.

        Parameters
        ----------
        parents : int
                  number of input individuals (-1 for any)
        offspring : int
                    number of output individuals
        """
        self.parents = parents
        self.offspring = offspring
        if offspring == 0:
            raise ValueError("Propagator has to sire more than 0 offspring.")
        return

    def __call__(self, inds):
        """
        Apply propagator.

        Parameters
        ----------
        inds: propulate.population.Individual
              individuals the propagator is applied to
        """
        raise NotImplementedError()


class Stochastic(Propagator):
    """
    Apply StochasticPropagator only with a given probability.

    If not applied the output still has to adhere to the defined number of offsprings.
    """
    def __init__(self, parents=0, offspring=0, probability=1.):
        """
        Constructor of StochasticPropagator class.

        Parameters
        ----------
        parents : int
                  number of input individuals (-1 for any)
        offspring : int
                    number of output individuals
        probability : float
                      probability of application
        """
        super(Stochastic, self).__init__(parents, offspring)
        self.probability = probability
        if offspring == 0:
            raise ValueError("Propagator has to sire more than 0 offspring.")
        return

    def __call__(self, inds):
        raise NotImplementedError()
        return


class Conditional(Propagator):
    """
    Apply different propagators depending on whether breeding population is complete or not.

    If population consists of less than the specified number of individuals required for breeding,
    a different propagator is applied than if this condition is fulfilled. 
    """
    def __init__(self, pop_size, true_prop, false_prop, parents=-1, offspring=-1):
        """
        Constructor of ConditionalPropagator class.

        Parameters
        ----------
        pop_size : int
                   breeding population size
        true_prop : propulate.propagators.Propagator
                    propagator applied if size of current population >= pop_size.
        false_prop : propulate.propagators.Propagator
                     propagator applied if size of current population < pop_size.
        parents : int
                  number of input individuals (-1 for any)
        offspring : int
                    number of output individuals
        """
        super(Conditional, self).__init__(parents, offspring)
        self.pop_size = pop_size
        self.true_prop = true_prop
        self.false_prop = false_prop

    def __call__(self, inds):
        if len(inds) >= self.pop_size:  # If number of evaluated individuals >= pop_size apply true_prop.
            return self.true_prop(inds)
        else:                           # Else apply false_prop.
            return self.false_prop(inds)


class Cascade(Propagator):

    def __init__(self, propagators):
        super(Cascade, self).__init__(propagators[0].parents, propagators[-1].offspring)
        self.propagators = propagators
        for i in range(len(propagators)-1):
            # Check compability of consecutive propagators in terms of number of parents + offsprings.
            if not _check_compatible(propagators[i].offspring, propagators[i+1].parents):
                outp = propagators[i]
                inp = propagators[i+1]
                outd = outp.offspring
                ind = indp.parents

                raise ValueError("Incompatible combination of {} output individuals of {} and {} input individuals of {}".format(outd, outp, ind, inp))

    def __call__(self, inds): # Apply propagators sequentially as requested in Cascade(...)
        for p in self.propagators:
            inds = p(inds)
        return inds


# TODO random number of points to mutate
class PointMutation(Stochastic):

    def __init__(self, limits, points=1, random=False, probability=1.):
        super(PointMutation, self).__init__(1, 1, probability)
        self.points = points
        self.limits = limits
        self.random = random
        if len(limits) < points:
            raise ValueError("Too many points to mutate for individual with {} traits".format(len(limits)))
        return

    def __call__(self, ind):
        """
        Apply point-mutation propagator.

        Parameters
        ----------
        ind: propulate.population.Individual
             individual the propagator is applied to
        """
        if random.random() < self.probability: # Apply propagator only with specified `probability` 
            ind = copy.deepcopy(ind)
            ind.loss = None # Initialize individual's loss attribute.
            # Determine traits to mutate via random sampling.
            # Return `self.points` length list of unique elements chosen from `ind.keys()`.
            # Used for random sampling without replacement.
            if self.random:
                self.points = random.randrange(len(self.limits))
            to_mutate = random.sample(ind.keys(), self.points)
            # Point-mutate `self.points` randomly chosen traits of individual `ind`.
            for i in to_mutate:
                if type(ind[i]) == int:
                    # Return randomly selected element from int range(start, stop, step).
                    ind[i] = random.randrange(*self.limits[i])
                elif type(ind[i]) == float:
                    # Return random floating point number N within limits.
                    ind[i] = random.uniform(*self.limits[i])
                elif type(ind[i]) == str:
                    # Return random element from non-empty sequence.
                    ind[i] = random.choice(self.limits[i])

        return ind # Return point-mutated individual.


# TODO rename to IntervalMutationClampedRelativeNormal? Or do this all in parameters if mu is set absolute and so on
class IntervalMutationNormal(Stochastic):

    def __init__(self, limits, sigma_factor=.1, points=1, probability=1.):
        super(IntervalMutationNormal, self).__init__(1, 1, probability)
        self.points = points # number of traits to point-mutate
        self.limits = limits
        self.sigma_factor = sigma_factor
        n_interval_traits = len([x for x in limits if type(limits[x][0]) == float])
        if n_interval_traits < points:
            raise ValueError("Too many points to mutate for individual with {} interval traits".format(n_interval_traits))
        return

    def __call__(self, ind):
        if random.random() < self.probability: # Apply propagator only with specified `probability`.
            ind = copy.deepcopy(ind)
            ind.loss = None # Initialize individual's loss attribute.
            # Determine traits of type float.
            interval_keys = [x for x in ind.keys() if type(ind[x])==float]
            # Determine ´self.points` traits to mutate.
            to_mutate = random.sample(interval_keys, self.points)
            # Mutate traits by sampling from Gaussian distribution centered around current value
            # with `sigma_factor` scaled interval width as standard distribution.
            for i in to_mutate:
                min_val, max_val = self.limits[i]           # Determine interval boundaries.
                mu = ind[i]                                 # Current value is mean.
                sigma = (max_val-min_val)*self.sigma_factor # Determine std from interval boundaries + sigma factor.
                ind[i] = random.gauss(mu, sigma)            # Sample new value from Gaussian blob centered around current value.
                ind[i] = min(max_val, ind[i])               # Make sure new value is within specified limits.
                ind[i] = max(min_val, ind[i])

        return ind # Return point-mutated individual.


class MateUniform(Stochastic): # uniform crossover

    def __init__(self, probability):
        super(MateUniform, self).__init__(2, 1, probability) # Breed 1 offspring from 2 parents.
        return

    def __call__(self, inds):
        ind = copy.deepcopy(inds[0]) # Consider 1st parent.
        if random.random() < self.probability: # Apply propagator only with specified `probability`.
            ind.loss = None # Initialize individual's loss attribute.
            # Replace traits in 1st parent with values of 2nd parent with a probability of 0.5.
            for k in inds[1].keys():
                if random.random() > 0.5:
                    ind[k] = inds[1][k]
        return ind # Return offspring.


class SelectBest(Propagator):
    """
    Select specified number of best performing individuals as evaluated by their losses.
    """
    def __init__(self, offspring):
        super(SelectBest, self).__init__(-1, offspring)
        return

    def __call__(self, inds):
        if len(inds) < self.offspring:
            raise ValueError("Has to have at least {} individuals to select the {} best ones.".format(self.offspring, self.offspring))
        # Sort elements of given iterable in specific order + return as list.
        return sorted(inds, key=lambda ind: ind.loss)[:self.offspring] # Return `self.offspring` best individuals in terms of loss.


class SelectUniform(Propagator):
    """
    Select specified number of individuals randomly.
    """
    def __init__(self, offspring):
        super(SelectUniform, self).__init__(-1, offspring)
        return

    def __call__(self, inds):
        if len(inds) < self.offspring:
            raise ValueError("Has to have at least {} individuals to select {} from them".format(self.offspring, self.offspring))
        # TODO sorted?
        # Return a `self.offspring` length list of unique elements chosen from `inds`. 
        # Used for random sampling without replacement.
        return random.sample(inds, self.offspring)


# TODO children != 1 case
# TODO parents should be fixed to one NOTE see utils reason why it is not right now
class InitUniform(Stochastic):
    """
    Initiliaze individuls by uniformly sampling specified limits for each trait.
    """
    def __init__(self, limits, parents=0, probability=1.):
        """
        In case of parents > 0 and probability < 1., call returns input individual without change.
        """
        super(InitUniform, self).__init__(parents, 1, probability)
        self.limits = limits
        return
    
    def __call__(self, *inds):
        if random.random() < self.probability: # Apply only with specified `probability`.
            ind = Individual() # Instantiate new individual.
            for limit in self.limits:
                # Randomly sample from specified limits for each trait.
                if type(self.limits[limit][0]) == int: # If ordinal trait of type integer.
                    ind[limit] = random.randrange(*self.limits[limit])
                elif type(self.limits[limit][0]) == float: # If interval trait of type float.
                    ind[limit] = random.uniform(*self.limits[limit])
                elif type(self.limits[limit][0]) == str: # If categorical trait of type string.
                    ind[limit] = random.choice(self.limits[limit])
                else:
                    raise ValueError("Unknown type of limits. Has to be float for interval, int for ordinal, or string for categorical.")
            return ind
        else:
            ind = inds[0]
            return ind # Return 1st input individual w/o changes.
