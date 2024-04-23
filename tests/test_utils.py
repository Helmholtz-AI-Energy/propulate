import pytest


@pytest.fixture(
    params=[
        ("rosenbrock", 0.0),
        ("step", -25.0),
        ("quartic", 0.0),
        ("rastrigin", 0.0),
        ("griewank", 0.0),
        ("schwefel", 0.0),
        ("bisphere", 0.0),
        ("birastrigin", 0.0),
        ("bukin", 0.0),
        ("eggcrate", -1.0),
        ("himmelblau", 0.0),
        ("keane", 0.6736675),
        ("leon", 0.0),
        ("sphere", 0.0),  # (fname, expected)
    ]
)
def function_parameters(request):
    """Define benchmark function parameter sets as used in tests."""
    return request.param
