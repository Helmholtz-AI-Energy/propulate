# Releasing a new version of Propulate

The current workflow for releasing a new version of `Propulate` is as follows:
1. Make sure the master branch is up-to-date and contains the version of the software that it is to be released.
2. On the master branch, update the version number in `setup.cfg`. We use semantic versioning.
3. Rebase release branch onto current master branch.
4. Make Github release from current master, including corresponding version tag.
5. Push release branch. This will trigger a Github action publishing the new release on PyPI.
