from random import Random
from typing import Dict, Tuple, List

import numpy as np

from ..propagators import Propagator, InitUniform, SelectMin, Gaussian
from ..population import Individual


class ReferenceNM(Propagator):
    def __init__(
        self,
        limits: Dict[str, Tuple[float, float]],
        rng: Random,
        alpha: float = 1.0,
        gamma: float = 2.0,
        rho: float = 0.5,
        sigma: float = 0.5,
    ):
        """ """
        super().__init__(limits, parents=-1, offspring=1, rng=rng)
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma

        self.step = "init"
        self.simplex = []
        self.centroid = None
        self.xr = None
        self.xe = None
        self.xc = None

        self.randominit = InitUniform(limits, rng=rng)
        self.initradius = 0.1

    def __call__(self, inds: List[Individual]) -> Individual:
        """ """
        # TODO remove evaluating centroid
        if len(inds) == 0:
            ind = self.randominit()
            self.step = "init"
        elif len(inds) < len(inds[0].position) + 1:
            self.simplex.append(inds[-1])
            offset = np.zeros_like(self.simplex[0].position)
            offset[len(self.simplex) - 1] = self.initradius
            position = self.simplex[0].position + offset
            ind = Individual(position, self.limits)

        elif len(inds) == len(self.simplex[0].position) + 1:
            self.simplex = [x for x in inds]
            # sort
            self.reset()
            ind = self.compute_centroid()
        else:
            newest = inds[-1]
            if self.centroid is None and self.step == "centroid":
                self.step = "reflect"
                self.centroid = newest
                # reflect
                ind = self.reflect()
                self.step = "reflect"
            elif self.xr is None and self.step == "reflect":
                self.xr = newest
                if (
                    self.simplex[0].loss <= self.xr.loss
                    and self.xr.loss < self.simplex[-2].loss
                ):
                    self.simplex[-1] = self.xr
                    self.reset()
                    ind = self.compute_centroid()
                elif self.xr.loss < self.simplex[0].loss:
                    ind = self.expand()  # expand
                    self.step = "expand"
                elif self.xr.loss >= self.simplex[-2].loss:
                    ind = self.contract()
                    self.step = "contract"
            elif self.xe is None and self.step == "expand":
                self.xe = newest
                if self.xe.loss < self.xr.loss:
                    self.simplex[-1] = self.xe
                    self.reset()
                    ind = self.compute_centroid()
                else:
                    self.simplex[-1] = self.xr
                    self.reset()
                    ind = self.compute_centroid()

            elif self.xc is None and self.step == "contract":
                self.xc = newest
                if self.xc.loss < self.xr.loss:
                    self.simplex[-1] = self.xc
                    self.reset()
                    ind = self.compute_centroid()

                else:
                    self.step = "shrink"
                    ind = self.shrink()
            elif self.step == "shrink":
                # TODO fix
                self.simplex[-1] = newest
                self.reset()
                ind = self.compute_centroid()

        return ind

    def compute_centroid(self):
        position = sum([x.position for x in self.simplex[:-1]]) / (
            len(self.simplex) - 1
        )

        return Individual(position, self.limits)

    def reflect(self):
        position = self.centroid.position + self.alpha * (
            self.centroid.position - self.simplex[-1].position
        )

        return Individual(position, self.limits)

    def expand(self):
        position = self.centroid.position + self.gamma * (
            self.xr.position - self.centroid.position
        )
        return Individual(position, self.limits)

    def contract(self):
        if self.xr.loss < self.simplex[-1].loss:
            position = self.centroid.position + self.rho * (
                self.xr.position - self.centroid.position
            )
        else:
            position = self.centroid.position + self.rho * (
                self.simplex[-1].position - self.centroid.position
            )
        return Individual(position, self.limits)

    def shrink(self):
        i = self.rng.randrange(1, len(self.simplex))

        position = self.simplex[i].position + self.sigma * (
            self.simplex[i].position - self.simplex[0].position
        )
        return Individual(position, self.limits)

    def reset(self):
        self.step = "centroid"
        self.simplex.sort(key=lambda x: x.loss)  # ascending
        self.centroid = None
        self.xr = None
        self.xe = None
        self.xc = None


class AdaptedNM(Propagator):
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
        """ """
        super().__init__(limits, parents=-1, offspring=1, rng=rng)
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
        self.select_simplex = SelectMin(self.limits, self.problem_dimension + 1)

    def __call__(self, inds: List[Individual]) -> Individual:
        """ """
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
                ind = self.contract()
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
        position = sum([x.position for x in self.simplex[:-1]]) / (
            len(self.simplex) - 1
        )

        return position

    def reflect(self):
        centroid = self.compute_centroid()
        position = centroid + self.alpha * (centroid - self.simplex[-1].position)

        return Individual(position, self.limits)

    def expand(self):
        centroid = self.compute_centroid()
        position = centroid + self.gamma * (self.simplex[0].position - centroid)
        return Individual(position, self.limits)

    def outercontract(self):
        centroid = self.compute_centroid()
        position = centroid + self.rho * (self.simplex[-1].position - centroid)
        return Individual(position, self.limits)

    def innercontract(self):
        # TODO find a way to make the distinction whether to expect a better point on the outside or the inside of the simplex
        # for now assume inside, if not the next generation should be a reflection, probably not the most efficient
        centroid = self.compute_centroid()
        position = centroid + self.rho * (self.simplex[-1].position - centroid)

        return Individual(position, self.limits)

    def shrink(self):
        i = self.rng.randrange(1, len(self.simplex))

        position = self.simplex[i].position + self.sigma * (
            self.simplex[i].position - self.simplex[0].position
        )
        return Individual(position, self.limits)
