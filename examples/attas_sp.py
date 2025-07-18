#!/usr/bin/env python3

"""ATTAS short-period motion."""

import argparse
import pathlib
import sys

import hedeut
import jax
import jax.flatten_util
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from scipy import optimize, sparse

from sidpax import common, sem


def program_args():
    parser = argparse.ArgumentParser(
        description=__doc__, fromfile_prefix_chars="@"
    )
    parser.add_argument(
        "--datafiles", type=pathlib.Path, nargs="*", help="Input data files."
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=200,
        help="Maximum number of optimizer iterations.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="File name base for saving the script output.",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Show plots interactively."
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Choose default datafiles if none selected
    if not args.output:
        script_path = pathlib.Path(sys.argv[0])
        out_dir = script_path.parent / "output"
        args.output = out_dir / script_path.with_suffix(".plot").name

    # Choose default datafiles if none selected
    if not args.datafiles:
        data_dir = pathlib.Path(sys.argv[0]).parent / "data"
        args.datafiles = [
            data_dir / "fAttasElv1.asc",
            data_dir / "fAttasElv2.asc",
        ]

    # Validate parameters
    assert all(f.exists() for f in args.datafiles), "Datafiles missing."

    # Return parsed arguments
    return args


class DimShortPeriod:
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
        return cls.Param()

    @hedeut.jax_vectorize_method(signature="(x),(u)->(x)", excluded={2})
    def fc(self, x, u, param):
        """Drift function."""
        # Unpack arguments
        alpha, q = x
        (dele,) = u

        # Unpack model parameters
        Z0 = param.Z0
        Za = param.Za
        Zq = param.Zq
        Zde = param.Zde
        M0 = param.M0
        Ma = param.Ma
        Mq = param.Mq
        Mde = param.Mde

        # Compute state derivatives
        alphadot = Z0 + Za * alpha + (Zq + 1) * q + Zde * dele
        qdot = M0 + Ma * alpha + Mq * q + Mde * dele

        # Assemble state derivative vector
        xdot = jnp.array([alphadot, qdot])
        return xdot

    def f(self, x, u, param):
        """Discrete-time state transition function."""
        return x + self.fc(x, u, param) * self.dt  # Euler's method

    @hedeut.jax_vectorize_method(signature="(x),(u)->(y)", excluded={2})
    def h(self, x, u, param):
        """Output function."""
        # Unpack arguments
        return x


if __name__ == "__main__":
    args = program_args()

    jax.config.update("jax_enable_x64", True)

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
    jac_coo = jax.jit(est.nres_jac_coo)

    def hess(v):
        jaci, jacj, jacv = jac_coo(data[0], unpack(v))
        J = sparse.coo_array((jacv, (jaci, jacj))).tocsc()
        return (J.T @ J).tocoo()
    
    from cyipopt import minimize_ipopt
    hess = jax.jit(jax.hessian(cost))
    result = minimize_ipopt(
        cost, paramvec, jac=grad, hess=hess, options=dict(disp=5, maxiter=100, linear_solver='ma57')
    )

    # hess = jax.jit(jax.hessian(cost))
    # result = optimize.minimize(
    #     cost, paramvec, method='trust-constr', jac=grad, hess=hess,
    #     options=dict(verbose=2, maxiter=10000)
    # )

    paramopt = unpack(result.x)
    yopt = est.model.h(paramopt.mu, data[0].u, paramopt.p)
