# Changelog

## Version 1.2.2

### What's Changed

#### New Features
* Enable support for *internally parallelized evaluation of the loss function using multi-rank workers* by @mcw92 in https://github.com/Helmholtz-AI-Energy/propulate/pull/110
* Provide example script for how to use Propulate with multi-rank workers for HPO / NAS of distributed data-parallel neural networks in `PyTorch` by @coquelin77 in https://github.com/Helmholtz-AI-Energy/propulate/pull/124
* Introduce *CMA-ES* propagator by @DonnyDevV in https://github.com/Helmholtz-AI-Energy/propulate/pull/91
* Introduce *particle swarm optimization* propagator by @Morridin in https://github.com/Helmholtz-AI-Energy/propulate/pull/78
* Introduce *Nelder-Mead* propagator by @oskar-taubert in https://github.com/Helmholtz-AI-Energy/propulate/pull/125
* Introduce *surrogate models* for early stopping of unpromising individuals by @vtotiv in https://github.com/Helmholtz-AI-Energy/propulate/pull/112
* Provide improved example script for how to use Propulate for HPO / NAS of neural networks in `PyTorch` (see `torch_example.py`) by @mcw92 in https://github.com/Helmholtz-AI-Energy/propulate/pull/106

#### Maintenance
* Introduce integration tests by @oskar-taubert in https://github.com/Helmholtz-AI-Energy/propulate/pull/98 and https://github.com/Helmholtz-AI-Energy/propulate/pull/99 and https://github.com/Helmholtz-AI-Energy/propulate/pull/116 and https://github.com/Helmholtz-AI-Energy/propulate/pull/140 and @mcw92 in https://github.com/Helmholtz-AI-Energy/propulate/pull/122
* Enable functional pre-commit hooks by @mcw92 in https://github.com/Helmholtz-AI-Energy/propulate/pull/114
* Improve tutorials and documentation by @mcw92 in https://github.com/Helmholtz-AI-Energy/propulate/pull/138


### New Contributors
* @DonnyDevV made their first contribution in https://github.com/Helmholtz-AI-Energy/propulate/pull/91
* @Morridin made their first contribution in https://github.com/Helmholtz-AI-Energy/propulate/pull/78
* @vtotiv made their first contribution in https://github.com/Helmholtz-AI-Energy/propulate/pull/112

**Full Changelog**: https://github.com/Helmholtz-AI-Energy/propulate/compare/v1.0.1...v1.2.2

## Version 1.1.0

### What's Changed

#### New Features
* Provide automatically built `Sphinx` _documentation including installation instructions, theoretical background, tutorials, and API references_ at https://propulate.readthedocs.io/ by @mcw92 in https://github.com/Helmholtz-AI-Energy/propulate/pull/71
* Add comprehensive tutorials
  * Add example script for how to use Propulate for HPO / NAS of neural networks in `PyTorch` by @oskar-taubert in https://github.com/Helmholtz-AI-Energy/propulate/pull/75
  * Iss12 - Add simple example script without islands by @mcw92 in https://github.com/Helmholtz-AI-Energy/propulate/pull/54
  * Include benchmark functions from Propulate publication by @mcw92 in https://github.com/Helmholtz-AI-Energy/propulate/pull/69
* Introduce _separate logging_ for `Propulate` optimizer, enabling using a separate logger within the loss function for, e.g., NAS by @coquelin77 in https://github.com/Helmholtz-AI-Energy/propulate/pull/73
* Introduce _clean and more intuitive checkpointing_
  * Iss44 - Create checkpoint path in `Pollinator` if not exists by @mcw92 in https://github.com/Helmholtz-AI-Energy/propulate/pull/51
* Add _contribution guidelines_ by @mcw92 in https://github.com/Helmholtz-AI-Energy/propulate/pull/79
* Add _typehints_ and clean and consistent _docstrings_ by @mcw92 in https://github.com/Helmholtz-AI-Energy/propulate/pull/64
* Iss40 - Boundary-inclusive sampling for ordinal parameters by @mcw92 in https://github.com/Helmholtz-AI-Energy/propulate/pull/52
* Iss46 - Introduce evaluation time as `Individual` attribute #46 by @SMEISEN in https://github.com/Helmholtz-AI-Energy/propulate/pull/47

#### Maintenance
* More cleaned-up, structured, and refactored code, including consistent docstrings, type hints, and meaningful names for classes and variables
  * Introduce `Propulator` base class by @mcw92 in https://github.com/Helmholtz-AI-Energy/propulate/pull/74
  * Iss39 - Remove inter-island communicator by @mcw92 in https://github.com/Helmholtz-AI-Energy/propulate/pull/53

### New Contributors
* @SMEISEN made their first contribution in https://github.com/Helmholtz-AI-Energy/propulate/pull/47

**Full Changelog**: https://github.com/Helmholtz-AI-Energy/propulate/compare/1.0.1...v1.1.0

## Version 1.0.1

### What's Changed
* Create first functional release by @oskar-taubert in https://github.com/Helmholtz-AI-Energy/propulate/pull/43 and https://github.com/Helmholtz-AI-Energy/propulate/pull/42

**Full Changelog**: https://github.com/Helmholtz-AI-Energy/propulate/compare/1.0.0...1.0.1

## Version 1.0.0
### What's Changed
#### New Features
* Provide new `lightning` example for HPO / NAS of neural networks in `PyTorch` by @oskar-taubert in https://github.com/Helmholtz-AI-Energy/propulate/pull/25

#### Maintenance
* Rename `SelectBest`/`SelectWorst` to `SelectMin`/`SelectMax` by @mcw92 in https://github.com/Helmholtz-AI-Energy/propulate/pull/27
* Rename wrapper to `islands` by @mcw92 in https://github.com/Helmholtz-AI-Energy/propulate/pull/29
* Create user-provided checkpoint folder if not exists by @mcw92 in https://github.com/Helmholtz-AI-Energy/propulate/pull/33

#### Bug fixes
* Fix `rng.sample()` call for Python 3.11 by @elcorto in https://github.com/Helmholtz-AI-Energy/propulate/pull/15
* Iss34 - Fix default propagator for ordinal-only HPO by @mcw92 in https://github.com/Helmholtz-AI-Energy/propulate/pull/37
* Iss38 - Fix continuing from checkpoint by @mcw92 in https://github.com/Helmholtz-AI-Energy/propulate/pull/41
* Fix checkpoint path by @oskar-taubert in https://github.com/Helmholtz-AI-Energy/propulate/pull/26

### New Contributors
* @elcorto made their first contribution in https://github.com/Helmholtz-AI-Energy/propulate/pull/15

**Full Changelog**: https://github.com/Helmholtz-AI-Energy/propulate/commits/1.0.0
