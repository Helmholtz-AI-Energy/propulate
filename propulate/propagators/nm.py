from random import Random
from typing import Dict, List, Tuple

import numpy as np

from ..population import Individual
from .base import Gaussian, Propagator, SelectMin


class ParallelNelderMead(Propagator):
    """
    Parallel Nelder Mead propagator.

    Attributes
    ----------
    limits : Dict[str, Tuple[float, float]] | Dict[str, Tuple[int, int]] | Dict[str, Tuple[str, ...]]
        The search space, i.e., the limits of (hyper-)parameters to be optimized.
    alpha: float
        reflection parameter
    gamma : float
        expansion parameter
    rho : float
        contraction parameter
    sigma : float
        shrinking parameter
    generation : int
        Current optimization iteration. Not the same as a normal Nelder Mead iteration, since there can only be a single loss evaluation per iteration in the adapted algorithm.
    simplex : list
        List of Individual instances forming the current simplex.

    """

    def __init__(
        self,
        limits: Dict[str, Tuple[float, float]],
        rng: Random,
        start: np.ndarray,
        alpha: float = 1.0,
        gamma: float = 2.0,
        rho: float = 0.5,
        sigma: float = 0.5,
        scale: float = 1.0,
    ):
        """
        Initialize Nelder-Mead propagator.

        Parameters
        ----------
        limits : Dict[str, Tuple[float, float]] | Dict[str, Tuple[int, int]] | Dict[str, Tuple[str, ...]]
            The search space, i.e., the limits of the (hyper-)parameters to be optimized.
        rng : random.Random, optional
            The separate random number generator of the Propulate optimization.
        start : numpy.ndarray
            Starting point around which the first simplex is constructed.
        alpha : float, optional
            Reflection parameter
        gamma : float, optional
            Expansion parameter
        rho : float, optional
            Contraction parameter
        sigma : float, optional
            Shrinking parameter
        scale: float, optional
            Size of the initial simplex.

        """
        super().__init__(parents=-1, offspring=1, rng=rng)
        self.limits = limits
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
        self.generation = 0
        self.simplex = None

        self.problem_dimension = len(start)
        self.start = start
        self.step = "init"

        self.init = Gaussian(
            limits,
            scale=scale,
            rng=np.random.default_rng(seed=self.rng.randint(0, 10000000)),
        )
        self.select_simplex = SelectMin(self.problem_dimension + 1)

    def __call__(self, inds: List[Individual]) -> Individual:
        """
        Apply the Nelder-Mead propagator.

        Parameters
        ----------
        inds : List[propulate.Individual]
            The individuals the propagator is applied to.

        Returns
        -------
        propulate.Individual
            The individual after application of the propagator.

        """
        if len(inds) < self.problem_dimension + 1:
            ind = self.init([Individual(self.start, self.limits)])
            ind.generation = self.generation
            self.step = "init"
        elif len(inds) == self.problem_dimension + 1:
            self.simplex = self.select_simplex(inds)
            ind = self.reflect()
            self.step = "reflect"
        else:
            self.simplex = self.select_simplex(inds)
            self.simplex.sort(key=lambda x: x.loss)
            if self.simplex[0].generation == self.generation:
                ind = self.expand()
                self.step = "expand"
            elif self.simplex[-1].generation == self.generation:
                ind = self.outercontract()
                self.step = "outercontract"
            elif self.generation not in set([x.generation for x in self.simplex]):
                if self.step == "expand":
                    ind = self.reflect()
                    self.step = "reflect"
                elif self.step == "reflect":
                    ind = self.outercontract()
                    self.step = "outercontract"
                elif self.step == "outercontract":
                    ind = self.innercontract()
                    self.step = "innercontract"
                elif self.step == "innercontract":
                    ind = self.shrink()
                    self.step = "shrink"
                else:
                    # TODO set scale depending on distance between best and centroid
                    ind = self.init([self.simplex[0]])
                    self.step = "desperation"

            else:
                ind = self.reflect()
                self.step = "reflect"

        self.generation += 1
        return ind

    def compute_centroid(self):
        """
        Compute the centroid of the current simplex.

        Returns
        -------
        numpy.ndarray
            Centroid of current simplex.
        """
        position = sum([x.position for x in self.simplex[:-1]]) / (
            len(self.simplex) - 1
        )
        return position

    def reflect(self):
        """
        Nelder-Mead reflect step.

        Returns
        -------
        numpy.ndarray
            Returns the reflection of the worst point in the simplex on the centroid.
        """
        centroid = self.compute_centroid()
        position = centroid + self.alpha * (centroid - self.simplex[-1].position)

        return Individual(position, self.limits)

    def expand(self):
        """
        Nelder-Mead expand step.

        Returns
        -------
        numpy.ndarray
            Returns the reflection of the worst point in the simplex on the centroid with a larger step size.
        """
        centroid = self.compute_centroid()
        position = centroid + self.gamma * (self.simplex[0].position - centroid)
        return Individual(position, self.limits)

    def outercontract(self):
        """
        Nelder-Mead contract step on the outside of the simplex.

        Returns
        -------
        numpy.ndarray
            Returns
        """
        centroid = self.compute_centroid()
        position = centroid + self.rho * (self.simplex[-1].position - centroid)
        return Individual(position, self.limits)

    def innercontract(self):
        """
        Nelder-Mead contract step on the inside of the simplex.

        Returns
        -------
        numpy.ndarray
        """
        # TODO find a way to make the distinction whether to expect a better point on the outside or the inside of the simplex
        # for now assume inside, if not the next generation should be a reflection, probably not the most efficient
        centroid = self.compute_centroid()
        position = centroid + self.rho * (self.simplex[-1].position - centroid)

        return Individual(position, self.limits)

    def shrink(self):
        """
        Nelder-Mead shrink step only touching a single random vertex of the current simplex.

        Returns
        -------
        numpy.ndarray
        """
        i = self.rng.randrange(1, len(self.simplex))

        position = self.simplex[i].position + self.sigma * (
            self.simplex[i].position - self.simplex[0].position
        )
        return Individual(position, self.limits)
