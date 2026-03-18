"""Output-Error Method."""

from dataclasses import dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import scipy.sparse

from sidpax.tree import merge_trees

from . import mat, sparse, stats
from .sparse import sparse_objective


@jdc.pytree_dataclass
class Data:
    y: jax.Array
    """Measured outputs."""

    u: jax.Array
    """Exogenous inputs."""

    def __len__(self):
        """Number of samples."""
        assert len(self.y) == len(self.u)
        return len(self.y)


@dataclass
class SegmentProblem:

    model: Any
    """Underlying dynamical system model."""

    @jdc.pytree_dataclass
    class Param:
        p: Any
        x0: jax.Array

    def param(self, data, rng=None):
        # Initialize parameters
        p = self.model.param(data, rng)
        x0 = jnp.zeros(self.model.nx)

        # Create dataclass and return
        return self.Param(p=p, x0=x0)

    def loglikelihood(self, param: Param, data: Data) -> jax.Array:
        model = self.model.bind(param.p)
        x_sim, y_sim = model.free_sim(param.x0, data.u)
        meas_logpdf = model.meas_logpdf(data.y, x_sim, data.u)
        return meas_logpdf.sum()


@dataclass
class Estimator:
    data: list[Data]
    subproblems: SegmentProblem | list[SegmentProblem]
    is_unique: SegmentProblem.Param = SegmentProblem.Param(p=True, x0=False)
    fix_p: bool = False

    def param(self, rng=None):
        param_list = [
            p.param(d, rng) for p, d in zip(self._subproblems, self.data)
        ]
        if self.fix_p:
            param_list = [jdc.replace(param, p={}) for param in param_list]
        return merge_trees(self.is_unique, *param_list)

    @property
    def _subproblems(self) -> list[SegmentProblem]:
        """A list with the subproblems."""
        if isinstance(self.subproblems, Sequence):
            return self.subproblems
        else:
            return [self.subproblems] * len(self.data)

    def cost(self, paramvec):
        params = self.unpack(paramvec)
        cost = 0.0
        for prob, param, data in zip(self._subproblems, params, self.data):
            cost = cost - prob.loglikelihood(param, data)
        return cost

    def grad(self, paramvec):
        return jax.grad(self.cost)(paramvec)

    def hessian(self, paramvec):
        return jax.hessian(self.cost)(paramvec)

    def unpack(self, paramvec):
        rng = jax.random.key(0)
        unpack = jax.flatten_util.ravel_pytree(self.param(rng))[1]
        return unpack(paramvec)
