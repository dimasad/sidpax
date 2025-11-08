#!/usr/bin/env python3

"""ATTAS short-period motion."""

import functools
import pathlib
import sys
from dataclasses import dataclass, field

import hedeut
import jax
import jax.flatten_util
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import tyro
from scipy import optimize, sparse

from sidpax import cli, common, mat, modeling, sem, stats


@dataclass
class CLIArguments:
    """Script command-line arguments."""

    @staticmethod
    def _datafiles_factory():
        data_dir = pathlib.Path(sys.argv[0]).parent / "data"
        return [
            data_dir / "fAttasElv1.asc",
            data_dir / "fAttasElv2.asc",
        ]

    @staticmethod
    def _output_factory():
        script_path = pathlib.Path(sys.argv[0])
        out_dir = script_path.parent / "output"
        return out_dir / script_path.with_suffix(".plot").name

    testing: cli.TestingArguments
    """Testing arguments."""

    jax: cli.JaxArguments
    """JAX configuration arguments."""

    datafiles: list[pathlib.Path] = field(default_factory=_datafiles_factory)
    """Input data files."""

    maxiter: int = 100
    """Maximum number of optimizer iterations."""

    output: pathlib.Path = field(default_factory=_output_factory)
    """File name base for saving the script output."""

    plot: bool = False
    """Show plots interactively."""

    def __post_init__(self):
        """Validate arguments and apply settings."""
        for f in self.datafiles:
            if not f.exists():
                raise ValueError(f"Datafile {f} does not exist.")

        if self.maxiter <= 0:
            raise ValueError("Maximum iterations must be positive.")


class DimShortPeriod(modeling.StateSpaceBase):
    """Dimensional short-period motion model."""

    nx: int = 2
    """Number of states."""

    nu: int = 1
    """Number of exogenous inputs."""

    ny: int = 2
    """Number of outputs."""

    dt: float = 0.04
    """Sampling period."""

    @jdc.pytree_dataclass
    class Param:
        Q: mat.PositiveDefiniteMatrix
        R: mat.PositiveDefiniteMatrix
        Z0: float = 0.0
        Za: float = 0.0
        Zq: float = 0.0
        Zde: float = 0.0
        M0: float = 0.0
        Ma: float = 0.0
        Mq: float = 0.0
        Mde: float = 0.0

    @classmethod
    def param(cls, data=None, rng=None):
        """Initialize the parameter structure."""
        Q = mat.mat.LExpDLT.identity(cls.nx)
        R = mat.mat.LExpDLT.identity(cls.ny)
        return cls.Param(Q=Q, R=R)

    @hedeut.jax_vectorize_method(signature="(x),(u)->(x)")
    def fc(self, x, u):
        """Drift function."""
        # Unpack arguments
        alpha, q = x
        (dele,) = u

        # Unpack model parameters
        Z0 = self.Z0
        Za = self.Za
        Zq = self.Zq
        Zde = self.Zde
        M0 = self.M0
        Ma = self.Ma
        Mq = self.Mq
        Mde = self.Mde

        # Compute state derivatives
        alphadot = Z0 + Za * alpha + (Zq + 1) * q + Zde * dele
        qdot = M0 + Ma * alpha + Mq * q + Mde * dele

        # Assemble state derivative vector
        xdot = jnp.array([alphadot, qdot])
        return xdot

    def f(self, x, u):
        """Discrete-time state transition function."""
        return x + self.fc(x, u) * self.dt  # Euler's method

    @common.jax_vectorize_method(signature="(x),(x),(u)->()")
    def trans_logpdf(self, xnext, x, u):
        """Log-density of a state transition, log p(x_{k+1} | x_k, u_k)."""
        return stats.mvn_logpdf(xnext, self.f(x, u), self.Q)

    @common.jax_vectorize_method(signature="(x),(u)->(y)")
    def h(self, x, u):
        """Output function."""
        return x

    @common.jax_vectorize_method(signature="(y),(x),(u)->()")
    def meas_logpdf(self, y, x, u):
        """Log-density of a measurement, log p(y_k | x_k, u_k)."""
        return stats.mvn_logpdf(y, self.h(x, u), self.R)


if __name__ == "__main__":
    args = tyro.cli(CLIArguments, description=__doc__)

    # Load Datafiles (all segments)
    d2r = np.pi / 180
    rawdata = [np.loadtxt(f.expanduser()) for f in args.datafiles]
    data = [None] * len(rawdata)
    for i, rawseg in enumerate(rawdata):
        y = jnp.c_[rawseg[:, 12] * d2r, rawseg[:, 7] * d2r]
        u = jnp.c_[rawseg[:, 21] * d2r]
        data[i] = sem.Data(y, u)
    dataest = data[:-1]
    dataval = data[-1]

    # Create the PRNG keys
    key = jax.random.key(0)
    key, init_key = jax.random.split(key)

    model = DimShortPeriod()
    est = sem.Estimator(model)
    param = est.param(dataest[0], init_key)
    paramvec, unpack = jax.flatten_util.ravel_pytree(param)

    cost = jax.jit(lambda v: est.cost(data[0], unpack(v)))
    grad = jax.jit(jax.grad(cost))
    hess_dense = jax.jit(jax.hessian(cost))
    hess_coo = jax.jit(lambda v: est.cost_hess(data[0], unpack(v)))
    def hess(v): return sparse.coo_array(hess_coo(v)).tocsc()
    hessp = jax.jit(lambda v, d: jax.jvp(
        grad, (v,), (jnp.astype(d, v.dtype),))[0])

    result = optimize.minimize(
        cost,
        paramvec,
        method="trust-constr",
        jac=grad,
        hess=hess,
        options=dict(verbose=2, maxiter=args.maxiter),
    )

    paramopt = unpack(jnp.astype(result.x, paramvec.dtype))
    popt = paramopt.p

    mdlopt = model.bind(paramopt.p)
    yopt = mdlopt.h(paramopt.mu, data[0].u)
