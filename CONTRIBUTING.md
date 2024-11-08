# Contributing to Propulate

Welcome to ``Propulate`` :dna:! We're thrilled that you're interested in contributing to our open-source project.
By participating, you can help improve the project and make it even better.

## How to Contribute

1. **Fork the Repository**: Click the "Fork" button at the top right corner of this repository's page to create your own copy.

2. **Clone Your Fork**: Clone your forked repository to your local machine using Git :octocat::
   ```bash
   git clone https://github.com/Helmholtz-AI-Energy/propulate.git
   ```

3. **Install the Package with Development Options** in a separate virtual environment from the main branch of your repo.
   This will put a number of pre-commit hooks for code linting and formatting with [Ruff](https://github.com/astral-sh/ruff)
   into place, ensuring PEP-8 conformity and overall good code quality consistently.
   The commands shown below work on Unix-based systems:
   ```bash
   python3 -m venv <insert/path/to/your/venv>
   source <insert/path/to/your/venv/bin/activate>
   python -m pip install -e ".[dev]"
   pre-commit install
   ```

4. **Create a Branch**: Create a new branch for your contribution. Choose a descriptive name. Depending on what you want
   to work on, prepend either of the following prefixes, `features`, `maintenance`, `bugfix`, or `hotfix`. Example:
   ```bash
   git checkout -b features/your-feature-name
   ```

5. **Make Changes**: Make your desired changes to the codebase. Please stick to the following guidelines:
   * `Propulate` uses [*Black*](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html) code style and so should you if you would like to contribute.
   * Please use type hints in all function definitions.
   * Please use American English for all comments and docstrings in the code.
   * `Propulate` uses Sphinx autoapi to automatically create API reference documentation from docstrings in the code.
     Please use the [NumPy Docstring Standard](https://numpydoc.readthedocs.io/en/latest/format.html) for your docstrings:

     ```python
     """
     Short Description

     Long Description (if needed)

     Parameters
     ----------
     param1 : type
         Description of param1.

     param2 : type, optional
         Description of param2. (if it's an optional argument)

     Returns
     -------
     return_type
         Description of the return value.

     Other Parameters
     ----------------
     param3 : type
         Description of param3. (if there are additional parameters)

     Raises
     ------
     ExceptionType
         Description of when and why this exception might be raised.

     See Also
     --------
     other_function : Related function or module.

     Examples
     --------
        >>> import numpy as np
        >>> x = np.array([1, 2, 3])
        >>> y = np.square(x)
        >>> print(y)
     array([1, 4, 9])

     Notes
     -----
     Additional notes, recommendations, or important information.
     """
     ```
     When applicable, please make references to parent modules and classes using ```:class:`ParentClassName` ```
as follows:

     ```python
     """
     This is the docstring for MyClass.

     Parameters
     ----------
     param1 : type
              Description of param1.

     Attributes
     ----------
     attr1 : type
         Description of attr1.

     See Also
     --------
     :class:`ParentClassName` : Reference to the parent class.

     """

     class ParentClassName:
         """
         The docstring for the parent class.
         """

     class MyClass(ParentClassName):
         """
         The docstring for MyClass.

         Parameters
         ----------
         param2 : type
                  Description of param2.

         Attributes
         ----------
         attr2 : type
                 Description of attr2.
         """
     ```
     In the example above, ``` :class:`ParentClassName` ``` is used to create a reference to the parent class `ParentClassName`.
     Sphinx autoapi will automatically generate links to the parent class documentation.


6. **Commit Changes**: Commit your changes with a clear and concise commit message:
   ```bash
   git commit -m "Add your commit message here"
   ```

7. **Push Changes**: Push your changes to your fork on GitHub:
   ```bash
   git push origin your-feature-name
   ```

8. **Rebase Onto Current Main:** Rebase your feature branch onto the current main branch of the original repo. Leaving
   this step out might lead to problems with the test workflow when merging your branch into the main later on. Rebasing
   will include any changes that might have been pushed into the main in the meantime and resolve possible conflicts.
   To sync your fork with the original upstream repo, check out [this page](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork)
   or follow the steps below. Note that before you can sync your fork with an upstream repo, you must configure a remote
   that points to the upstream repository in Git.
   ```
   cd <path/to/your/local/project/fork>
   git fetch upstream
   git checkout main
   git merge upstream/main
   git rebase main features/your-feature-name
   ```

9. **Open a Pull Request**: Go to the [original repository](https://github.com/Helmholtz-AI-Energy/propulate.git) and click the "New Pull Request" button. Follow the guidelines in the template to submit your pull request.

## Code of Conduct

Please note that we have a [Code of Conduct](CODE_OF_CONDUCT.md), and we expect all contributors to follow it. Be kind and respectful to one another.

## Questions or Issues

If you have questions or encounter any issues, please create an issue in the [Issues](https://github.com/Helmholtz-AI-Energy/propulate/issues) section of this repository.

Thank you for your contribution!
