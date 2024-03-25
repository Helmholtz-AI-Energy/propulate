import random
import GPy
from mpi4py import MPI
import numpy as np
from typing import Tuple, Dict, Union
from propulate import Surrogate
from propulate.population import Individual


# Gaussian Process Regression with an Exponential Decay Kernel Surrogate
class DynamicSurrogate(Surrogate):
    """
    Surrogate model using a two-step Gaussian Process Regression
    to predict the final loss of a configuration.
    The first global model is trained on the configurations with their final losses.
    The second local model is trained on the interim losses of the current run.
    The local model uses the global model's prediction as the mean function.

    Loosely based on the paper:
    Freeze-Thaw Bayesian Optimization
    https://arxiv.org/abs/1406.3896

    Attributes
    ----------
    limits : Union[Dict[str, Tuple[float, float]], Dict[str, Tuple[int, int]], Dict[str, Tuple[str, ...]]]
        The hyperparameter configuration space's limits.
    encodings : Dict[str, Dict[str, int]]
        The encoding of categorical limit values to int.
    history_X : np.ndarray
        The encoded configurations.
    history_Y : np.ndarray
        The final losses corresponding to the encoded configurations.
    current_run_data : np.ndarray
        An array containing the interim losses of the current run.
    current_encoding : np.ndarray
        The encoded configuration of the current run.
    global_kernel : GPy.kern.Kern
        The kernel for the global Gaussian Process Regression model.
    local_kernel : GPy.kern.Kern
        The kernel for the local Gaussian Process Regression model.
    mean_function : GPy.mappings.Mapping
        The calculated mean function for the local model.
    global_gpr : GPy.models.GPRegression
        The global Gaussian Process Regression model.
    local_gpr : GPy.models.GPRegression
        The local Gaussian Process Regression model.
    first_run : bool
        A boolean indicating if this is the first run.
    max_idx : int
        The maximum index, i.e. the number of interim loss values for every run.
    allowed_loss_margin : float
        The allowed loss margin for cancelling a run.

    Methods
    -------
    __init__()
        Override the parent class's constructor to include the configuration space limits.
    start_run()
        Initialize the current run. Create the mean function for the local model.
    update()
        Update the global gaussian process model with the final loss of the current run.
    cancel()
        Cancel the current run if the local model predicts a higher final loss than the best so far.
    merge()
        Merge a configuration and loss tuple into the history arrays.
    data()
        Return the latest configuration and loss tuple from the history arrays.


    Notes
    -----
    The ``DynamicSurrogate`` class implements all methods from the ``Surrogate`` class.

    See Also
    --------
    :class:`Surrogate` : The parent class.
    """

    def __init__(
        self,
        limits: Union[
            Dict[str, Tuple[float, float]],
            Dict[str, Tuple[int, int]],
            Dict[str, Tuple[str, ...]],
        ],
    ) -> None:
        """
        Initialize the DynamicSurrogate with the configuration space limits.
        Set the global and local kernels for Gaussian Process Regression.
        All other needed attributes are initialized as empty arrays or None.

        Parameters
        ----------
        limits : Union[Dict[str, Tuple[float, float]], Dict[str, Tuple[int, int]], Dict[str, Tuple[str, ...]]
            The hyperparameter configuration space's limits.
        """
        self.limits = limits
        self.encodings = self._create_encoding(limits)

        # History arrays to store (encoded configuration, final loss) pairs
        self.history_X: np.ndarray = np.array([[]])
        self.history_Y: np.ndarray = np.array([[]])
        # Arrays to store interim loss for the current run
        self.current_run_data: np.ndarray = np.array([[]])
        self.current_encoding: np.ndarray = np.array([[]])

        self.global_kernel = GPy.kern.Matern32(input_dim=len(limits)) + GPy.kern.White(
            input_dim=len(limits)
        )
        self.local_kernel = GPy.kern.Matern52(
            1, variance=1.0, lengthscale=1.0, ARD=True
        ) + GPy.kern.White(1, variance=1e-5)

        self.mean_function = None

        # Mean loss
        self.global_gpr: GPy.models.GPRegression = None
        # Newly created for every run
        self.local_gpr: GPy.models.GPRegression = None

        self.first_run: bool = True
        self.max_idx: int = 0

        self.allowed_loss_margin: float = 0.8

        # Set seed for reproducibility.
        rank = MPI.COMM_WORLD.Get_rank()
        np.random.seed(42 * rank)
        random.seed(42 * rank)

    def start_run(self, ind: Individual) -> None:
        """
        Encode the configuration given as individual and create the mean function
        for the local model by using the global model's prediction.

        Parameters
        ----------
        ind : Individual
            The individual containing the current configuration.
        """
        self.current_encoding = self.encode_configuration(ind)

        if self.first_run:
            return

        # Use global model's prediction as the mean function for the local model.
        self.mean_function = GPy.mappings.Constant(input_dim=1, output_dim=1)
        mean, variance = self.global_gpr.predict(self.current_encoding)

        self.mean_function.C = mean[0]

        # Reset the local model and run data.
        self.local_gpr = None
        self.current_run_data = np.array([[]])

    def update(self, loss: float) -> None:
        """
        Update the model with the final loss of the current run
        and retrain the global Gaussian Process with the new data.

        Parameters
        ----------
        loss : float
            The final loss of the current run.
        """
        # Append the final loss to the history.
        # Only if this is the first run:
        if self.first_run:
            if self.history_X.size == 0:
                self.history_X = self.current_encoding
            else:
                self.history_X = np.vstack([self.history_X, self.current_encoding])
            if self.history_Y.size == 0:
                self.history_Y = np.array([[loss]])
            else:
                self.history_Y = np.vstack([self.history_Y, [loss]])

            # Create global model and fit with the new data.
            self.global_gpr = GPy.models.GPRegression(
                self.history_X, self.history_Y, self.global_kernel
            )
            self.global_gpr.optimize()

            self.first_run = False
            return

        # Or append if the run is finished.
        if self.max_idx == len(self.current_encoding):
            if self.history_X.size == 0:
                self.history_X = self.current_encoding
            else:
                self.history_X = np.vstack([self.history_X, self.current_encoding])
            if self.history_Y.size == 0:
                self.history_Y = np.array([[loss]])
            else:
                self.history_Y = np.vstack([self.history_Y, [loss]])

            # Fit global model with the new data.
            self.global_gpr.set_XY(self.history_X, self.history_Y)
            self.global_gpr.optimize()
            return

    def cancel(self, loss: float) -> bool:
        """
        Cancel the current run if the local Gaussian Process predicts a final loss
        that is higher than the best known loss with an allowed margin.

        Parameters
        ----------
        loss : float
            The next interim loss of the current run.

        Returns
        -------
        bool
            True if the run should be cancelled, False otherwise.
        """
        # The first run is never cancelled and has to run till the end.
        if self.first_run:
            # Here we also count how often cancel is called for later reference.
            self.max_idx += 1
            return False

        # Append interim loss to the current run data.
        if self.current_run_data.size == 0:
            self.current_run_data = np.array([[loss]])
        else:
            self.current_run_data = np.vstack([self.current_run_data, [loss]])

        if self.local_gpr is None:
            self.local_gpr = GPy.models.GPRegression(
                np.arange(1, self.current_run_data.shape[0] + 1).reshape(-1, 1),
                self.current_run_data,
                self.local_kernel,
                mean_function=self.mean_function,
            )
        else:
            # Retrain the local model with X consisting of synthetic indices.
            self.local_gpr.set_XY(
                np.arange(1, self.current_run_data.shape[0] + 1).reshape(-1, 1),
                self.current_run_data,
            )

        self.local_gpr.optimize()

        # Predict the final loss.
        final_loss, final_variance = self.local_gpr.predict(np.array([[self.max_idx]]))
        # Check if the final loss is lower than the best loss with margin so far.
        if ((final_loss - final_variance) * self.allowed_loss_margin) > max(
            self.history_Y
        )[0]:
            return True

        return False

    def merge(self, data: Tuple[np.ndarray, float]) -> None:
        """
        Merge a configuration and loss tuple into the history arrays.
        Then retrain the global Gaussian Process with the new data.

        Parameters
        ----------
        data : Tuple[np.ndarray, float]
            The configurations and final losses of the merged run.
        """
        if len(data) != 2:
            raise ValueError("Data must be a tuple of (configuration, loss)")
        if data[1] == 0:
            # Ignore 0 loss as this means some kind of unexpected behavior.
            return

        # Append new data loss to the existing.
        if self.history_X.size == 0:
            self.history_X = np.array([data[0]])
        else:
            self.history_X = np.vstack([self.history_X, [data[0]]])
        if self.history_Y.size == 0:
            self.history_Y = np.array([data[1]])
        else:
            self.history_Y = np.vstack([self.history_Y, [data[1]]])

        # Fit global model with the new data.
        if self.global_gpr is None:
            self.global_gpr = GPy.models.GPRegression(
                self.history_X, self.history_Y, self.global_kernel
            )
        else:
            self.global_gpr.set_XY(self.history_X, self.history_Y)

        self.global_gpr.optimize()

    def data(self) -> Tuple[np.ndarray, float]:
        """
        Return the latest configuration and loss from the history arrays.

        Returns
        -------
        Tuple[np.ndarray, float]
            The latest configuration and loss.
        """
        # Return empty array if no data is available.
        if self.history_X.size == 0 or self.history_Y.size == 0:
            return (np.array([]), 0)

        # Return the latest loss value as all other values are already shared.
        return (self.history_X[-1], self.history_Y[-1])

    def _create_encoding(
        self,
        limits: Union[
            Dict[str, Tuple[float, float]],
            Dict[str, Tuple[int, int]],
            Dict[str, Tuple[str, ...]],
        ],
    ) -> Dict[str, Dict[str, int]]:
        """
        Create a mapping from categorical values to integers.
        Gaussian Process Regression only accepts numerical values.

        Parameters
        ----------
        limits : Union[Dict[str, Tuple[float, float]], Dict[str, Tuple[int, int]], Dict[str, Tuple[str, ...]]
            The hyperparameter configuration space's limits.

        Returns
        -------
        Dict[str, Dict[str, int]]
            The input dict but with the categorical values transformed to integers.
        """
        encodings = {}
        for key, values in limits.items():
            if isinstance(values, tuple) and isinstance(values[0], str):
                # create a mapping for categorical values
                encodings[key] = {v: i for i, v in enumerate(values)}
        return encodings

    def encode_configuration(
        self,
        config: Union[
            Dict[str, float],
            Dict[str, int],
            Dict[str, str],
        ],
    ) -> np.ndarray:
        """
        Encode a configuration dictionary into a 2D array.
        This is the format required by the Gaussian Process Regression model.

        Parameters
        ----------
        config : Union[Dict[str, float], Dict[str, int], Dict[str, str]]
            The configuration dictionary to encode.

        Returns
        -------
        np.ndarray
            The encoded configuration.
        """
        # Initialize an empty array to ``limits`` size.
        encoded_array = np.zeros((1, len(self.limits)), dtype=float)

        # Fill the array with encoded values.
        for i, key in enumerate(self.limits):
            value = config[key]
            if key in self.encodings:
                # Ignore type hint as value can only be a string.
                encoded_array[0, i] = self.encodings[key][value]
            else:
                encoded_array[0, i] = value

        return encoded_array
