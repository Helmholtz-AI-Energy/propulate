from random import Random
from typing import Dict, Tuple, List

import numpy as np

from ..propagators import Propagator, InitUniform
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

        self.step = 0
        self.simplex = []
        self.centroid = None
        self.xr = None
        self.xe = None
        self.xc = None
        self.to_shrink = []

        self.randominit = InitUniform(limits, rng=rng)
        self.initradius = 0.1

    def __call__(self, inds: List[Individual]) -> Individual:
        """ """
        print(self.step)
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
            print(newest.loss)
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
