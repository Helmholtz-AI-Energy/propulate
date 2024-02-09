import random
import GPy
from mpi4py import MPI
import numpy as np
from typing import Tuple, Dict, Union
from propulate import Surrogate
from propulate.population import Individual


# Gaussian Process Regression with an Exponential Decay Kernel Surrogate
class DynamicSurrogate(Surrogate):
    def __init__(
        self,
        limits: Union[
            Dict[str, Tuple[float, float]],
            Dict[str, Tuple[int, int]],
            Dict[str, Tuple[str, ...]],
        ],
    ) -> None:
        print("Dynamic Surrogate - init")
        self.limits = limits
        self.encodings = self._create_encoding(limits)

        # to store (encoded configuration, final loss) pairs
        self.history_X: np.ndarray = np.array([[]])
        self.history_Y: np.ndarray = np.array([[]])
        # to store interim loss for the current run
        self.current_run_data: np.ndarray = np.array([[]])
        self.current_encoding: np.ndarray = np.array([[]])

        self.global_kernel = GPy.kern.Matern32(input_dim=len(limits)) + GPy.kern.White(input_dim=len(limits))
        self.local_kernel = GPy.kern.Matern52(1, variance=1., lengthscale=1., ARD=True) + GPy.kern.White(1, variance=1e-5)

        self.mean_function = None

        # mean loss
        self.global_gpr: GPy.models.GPRegression = None
        # newly created for every run
        self.local_gpr: GPy.models.GPRegression = None

        self.first_run: bool = True
        self.max_idx: int = 0

        self.allowed_loss_margin: float = 0.8

        # Set seed for reproducibility
        rank = MPI.COMM_WORLD.Get_rank()
        np.random.seed(42 * rank)
        random.seed(42 * rank)

    def start_run(self, ind: Individual):
        print("Dynamic Surrogate - start run")
        self.current_encoding = self.encode_configuration(ind)
        print("Dynamic Surrogate - start run - encoding", self.current_encoding)

        if self.first_run:
            return

        # use global model's prediction as the mean function for the local model
        self.mean_function = GPy.mappings.Constant(input_dim=1, output_dim=1)
        mean, variance = self.global_gpr.predict(self.current_encoding)
        print("Dynamic Surrogate - start run - mean and variance", mean, variance)
        self.mean_function.C = mean[0]

        # reset the local model and run data
        self.local_gpr = None
        self.current_run_data = np.array([[]])

    def update(self, loss: float) -> None:
        print("Dynamic Surrogate - update")
        # append the final loss to the history
        # only if this is the first run
        if self.first_run:
            if self.history_X.size == 0:
                self.history_X = self.current_encoding
            else:
                self.history_X = np.vstack([self.history_X, self.current_encoding])
            if self.history_Y.size == 0:
                self.history_Y = np.array([[loss]])
            else:
                self.history_Y = np.vstack([self.history_Y, [loss]])

            # create global model and fit with the new data
            self.global_gpr = GPy.models.GPRegression(self.history_X, self.history_Y, self.global_kernel)
            self.global_gpr.optimize()

            self.first_run = False
            return

        # or append if the run is finished
        if self.max_idx == len(self.current_encoding):
            if self.history_X.size == 0:
                self.history_X = self.current_encoding
            else:
                self.history_X = np.vstack([self.history_X, self.current_encoding])
            if self.history_Y.size == 0:
                self.history_Y = np.array([[loss]])
            else:
                self.history_Y = np.vstack([self.history_Y, [loss]])

            # fit global model with the new data
            self.global_gpr.set_XY(self.history_X, self.history_Y)
            self.global_gpr.optimize()
            return

    def cancel(self, loss: float) -> bool:
        print("Dynamic Surrogate - cancel - loss", loss)
        # the first run is never cancelled and has to run till the end
        if self.first_run:
            # here we also count how often cancel is called for later reference
            self.max_idx += 1
            return False

        # append interim loss to the current run data
        if self.current_run_data.size == 0:
            self.current_run_data = np.array([[loss]])
        else:
            self.current_run_data = np.vstack([self.current_run_data, [loss]])

        if self.local_gpr is None:
            self.local_gpr = GPy.models.GPRegression(
                np.arange(1, self.current_run_data.shape[0] + 1).reshape(-1, 1),
                self.current_run_data,
                self.local_kernel,
                mean_function=self.mean_function
            )
        else:
            # retrain the local model with X consisting of synthetic indices
            self.local_gpr.set_XY(
                np.arange(1, self.current_run_data.shape[0] + 1).reshape(-1, 1),
                self.current_run_data,
            )

        self.local_gpr.optimize()

        # predict the final loss
        final_loss, final_variance = self.local_gpr.predict(np.array([[self.max_idx]]))
        print("Dynamic Surrogate - cancel - final loss", final_loss, "final variance", final_variance)
        # check if the final loss is lower than the best loss with margin so far
        if ((final_loss - final_variance) * self.allowed_loss_margin) > max(self.history_Y)[0]:
            print("Dynamic Surrogate - cancel - final loss",
                  (final_loss - final_variance) * self.allowed_loss_margin,
                  "is lower than the best loss with margin so far",
                  max(self.history_Y)[0])
            return True

        return False

    def merge(self, data: Tuple[np.ndarray, float]) -> None:
        print("Dynamic Surrogate - merge")
        if len(data) != 2:
            raise ValueError("Data must be a tuple of (configuration, loss)")
        if data[1] == 0:
            # ignore 0 loss as this means some kind of unexpected behaviour
            return

        # append new data loss to the existing
        if self.history_X.size == 0:
            self.history_X = np.array([data[0]])
        else:
            self.history_X = np.vstack([self.history_X, [data[0]]])
        if self.history_Y.size == 0:
            self.history_Y = np.array([data[1]])
        else:
            self.history_Y = np.vstack([self.history_Y, [data[1]]])

        # fit global model with the new data
        if self.global_gpr is None:
            self.global_gpr = GPy.models.GPRegression(
                    self.history_X, self.history_Y, self.global_kernel)
        else:
            self.global_gpr.set_XY(self.history_X, self.history_Y)

        self.global_gpr.optimize()

    def data(self) -> Tuple[np.ndarray, float]:
        print("Dynamic Surrogate - data")
        # return empty array if no data is available
        if self.history_X.size == 0 or self.history_Y.size == 0:
            return (np.array([]), 0)

        # return the latest loss value
        # as all other values are already shared
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
        """
        # initialize an empty array to limits size
        encoded_array = np.zeros((1, len(self.limits)), dtype=float)

        # fill the array with encoded values
        for i, key in enumerate(self.limits):
            value = config[key]
            if key in self.encodings:
                # ignore type hint as value can only be a string
                encoded_array[0, i] = self.encodings[key][value]
            else:
                encoded_array[0, i] = value

        return encoded_array
