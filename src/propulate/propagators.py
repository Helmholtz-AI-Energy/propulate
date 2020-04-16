import random
import copy

# TODO develop a zoo of propagators that can be assembled by user akin to torch dataset transforms

def mutate(ind, limits):
    individual = copy.deepcopy(ind)
    to_mutate = random.choice(list(individual.keys()))
    if type(individual[to_mutate]) == int:
        individual[to_mutate] = random.randrange(*limits[to_mutate])
    elif type(individual[to_mutate]) == float:
        individual[to_mutate] = random.uniform(*limits[to_mutate])
    elif type(individual[to_mutate]) == str:
        individual[to_mutate] = random.choice(limits[to_mutate])

    return individual

# TODO add boltzmann mating (or whatever they would call it)
def mate(parent1, parent2):
    ind = copy.deepcopy(parent1)

    for k in parent2.keys():
        if random.random() > 0.5:
            ind[k] = parent2[k]
    return ind

