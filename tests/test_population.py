import pytest

from propulate.population import Individual


@pytest.mark.mpi_skip
def test_individual():
    """Test the mapping between dictionary and embedded vector representation of Individuals."""
    limits = {
        "float1": (0.0, 1.0),
        "float2": (-1.0, 1.0),
        "int1": (0, 5),
        "int2": (1, 8),
        "cat1": ("a", "b", "c", "d", "e"),
        "cat2": ("f", "g", "h"),
    }
    ind_map = {
        "float1": 0.1,
        "float2": 0.2,
        "int1": 3,
        "int2": 4,
        "cat1": "e",
        "cat2": "f",
    }

    ind = Individual(ind_map, limits)
    assert len(ind) == 6
    assert ind.position.shape[0] == 12

    assert ind["cat1"] == "e"
    assert ind["cat2"] == "f"

    ind["cat1"] = "b"
    assert ind.position[5] == 1.0
    assert ind.position[9] == 1.0
    assert list(ind.keys()) == list(ind_map.keys())


@pytest.mark.mpi_skip
def test_individual_order():
    """Test the mapping between dictionary and embedded vector representation of Individuals at different ordering."""
    limits = {
        "float1": (0.0, 1.0),
        "cat1": ("a", "b", "c", "d", "e"),
        "float2": (-1.0, 1.0),
        "int1": (0, 5),
        "cat2": ("f", "g", "h"),
        "int2": (1, 8),
    }
    ind_map = {
        "cat1": "e",
        "cat2": "f",
        "float1": 0.1,
        "float2": 0.2,
        "int1": 3,
        "int2": 4,
    }

    ind = Individual(ind_map, limits)
    assert len(ind) == 6
    assert ind.position.shape[0] == 12

    assert ind["cat1"] == "e"
    assert ind["cat2"] == "f"

    ind["cat1"] = "b"
    assert ind.position[1] == 0.0
    assert ind.position[2] == 1.0
    assert ind.position[8] == 1.0
    assert ind.position[9] == 0.0


@pytest.mark.mpi_skip
def test_special_key():
    """Test that using _ prefixed keys does not trigger the categorical embedding correctly."""
    limits = {
        "float1": (0.0, 1.0),
        "float2": (-1.0, 1.0),
        "int1": (0, 5),
        "int2": (1, 8),
        "cat1": ("a", "b", "c", "d", "e"),
        "cat2": ("f", "g", "h"),
    }
    ind_map = {
        "float1": 0.1,
        "float2": 0.2,
        "int1": 3,
        "int2": 4,
        "cat1": "e",
        "cat2": "f",
    }

    ind = Individual(ind_map, limits)
    assert len(ind) == 6
    assert ind.position.shape[0] == 12

    assert ind["cat1"] == "e"
    assert ind["cat2"] == "f"

    ind["cat1"] = "b"
    ind["_s"] = [1, 0.2, 3]
    assert ind.position[5] == 1.0
