from propulate.population import Individual


def test_individual():
    ind_map = {
        "float1": 0.1,
        "float2": 0.2,
        "int1": 3,
        "int2": 4,
        "cat1": "e",
        "cat2": "f",
    }

    ind = Individual(ind_map)
    print(ind)


def test_limits():
    raise
