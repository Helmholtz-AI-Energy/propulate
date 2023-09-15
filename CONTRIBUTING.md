# Contributing to Propulate

Welcome to ``Propulate``! We're thrilled that you're interested in contributing to our open-source project. 
By participating, you can help improve the project and make it even better. 

## How to Contribute

1. **Fork the Repository**: Click the "Fork" button at the top right corner of this repository's page to create your own copy.

2. **Clone Your Fork**: Clone your forked repository to your local machine using Git:
   ```bash
   git clone https://github.com/Helmholtz-AI-Energy/propulate.git
   ```

3. **Create a Branch**: Create a new branch for your contribution. Choose a descriptive name:
   ```bash
   git checkout -b your-feature-name
   ```

4. **Make Changes**: Make your desired changes to the codebase. Please stick to the following guidelines: 
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
   
        
5. **Commit Changes**: Commit your changes with a clear and concise commit message:
   ```bash
   git commit -m "Add your commit message here"
   ```

6. **Push Changes**: Push your changes to your fork on GitHub:
   ```bash
   git push origin your-feature-name
   ```

7. **Open a Pull Request**: Go to the [original repository](https://github.com/Helmholtz-AI-Energy/propulate.git) and click the "New Pull Request" button. Follow the guidelines in the template to submit your pull request.

## Code of Conduct

Please note that we have a [Code of Conduct](CODE_OF_CONDUCT.md), and we expect all contributors to follow it. Be kind and respectful to one another.

## Questions or Issues

If you have questions or encounter any issues, please create an issue in the [Issues](https://github.com/Helmholtz-AI-Energy/propulate/issues) section of this repository.

Thank you for your contribution!