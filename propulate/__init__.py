from importlib.metadata import PackageNotFoundError, version

try:
    # Change here if project is renamed and does not equal the package name
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from . import propagators
from .islands import Islands
from .migrator import Migrator
from .pollinator import Pollinator
from .population import Individual
from .propulator import Propulator
from .surrogate import Surrogate
from .utils import get_default_propagator, set_logger_config

__all__ = [
    "Islands",
    "Individual",
    "Propulator",
    "Surrogate",
    "Migrator",
    "Pollinator",
    "get_default_propagator",
    "set_logger_config",
    "propagators",
]
