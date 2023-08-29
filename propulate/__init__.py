# -*- coding: utf-8 -*-
from pkg_resources import DistributionNotFound, get_distribution

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound

from .islands import Islands
from .propulator import Propulator
from .migrator import Migrator
from .pollinator import Pollinator
from .utils import get_default_propagator, set_logger_config

__all__ = [
    "Islands",
    "Propulator",
    "Migrator",
    "Pollinator",
    "get_default_propagator",
    "set_logger_config",
]
