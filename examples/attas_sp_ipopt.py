#!/usr/bin/env python3

"""ATTAS short-period motion, Variational System Identification.

Based on data and code (test case 11) from
"Flight Vehicle System Identification - A Time Domain Methodology"
Second Edition
Author: Ravindra V. Jategaonkar
Published by AIAA, Reston, VA 20191, USA

Data available at
    https://arc.aiaa.org/doi/suppl/10.2514/4.102790/suppl_file/flt_data.zip

Original code available at
    https://arc.aiaa.org/doi/suppl/10.2514/4.102790/suppl_file/chapter04.zip

An earlier version of this script was published in the repository
https://github.com/dimasad/scitech-2025-code/ and the paper "Variational System
Identification of Aircraft", presented in AIAA SciTech 2025,
[DOI:10.2514/6.2025-1253](https://arc.aiaa.org/doi/10.2514/6.2025-1253) and
[arXiv:2510.26496](https://arxiv.org/abs/2510.26496).
"""

import pathlib
import sys
from dataclasses import dataclass, field
from typing import Literal

import cyipopt
import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy as jsp
import jax_dataclasses as jdc
import numpy as np
import tyro
from scipy import optimize, sparse

from sidpax import cli, common, mat, sem
from sidpax.modeling import EulerDiscretization, MVNMeasurement, MVNTransition


@dataclass
class CLIArguments:
    """Script command-line arguments."""

    @dataclass
    class JaxArguments(cli.JaxArguments):
        """JAX configuration arguments."""

        x64: bool = True
        """Use double precision (64bits) in JAX."""

        platform: Literal["auto", "cpu", "gpu"] = "cpu"

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

    jax: JaxArguments
    """JAX configuration arguments."""

    datafiles: list[pathlib.Path] = field(default_factory=_datafiles_factory)
    """Input data files."""

    maxiter: int = 500
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


class DimShortPeriod(MVNMeasurement, MVNTransition, EulerDiscretization):
    """Dimensional short-period motion model."""

    nx: int = 2
    """Number of states."""

    nu: int = 1
    """Number of exogenous inputs."""

    ny: int = 3
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
        az0: float = 0.0
        V: float = 0.0

    @classmethod
    def param(cls, data=None, rng=None):
        """Initialize the parameter structure."""
        Q = mat.LExpDLT.identity(cls.nx)
        R = mat.LExpDLT.identity(cls.ny)
        return cls.Param(Q=Q, R=R)

    @common.jax_vectorize_method(signature="(x),(u)->(x)")
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

    @common.jax_vectorize_method(signature="(x),(u)->(y)")
    def h(self, x, u):
        """Output function."""
        # Unpack arguments
        alpha, q = x
        (dele,) = u

        # Unpack model parameters
        Z0 = self.Z0
        Za = self.Za
        Zq = self.Zq
        Zde = self.Zde
        az0 = self.az0
        V = self.V

        # Compute alphadot
        alphadot = Z0 + Za * alpha + (Zq + 1) * q + Zde * dele
        az = V * (alphadot - q) + az0

        # Assemble output vector and return
        return jnp.array([alpha, q, az])

    def prior_logpdf(self, x0):
        """Prior log-density of the initial state and parameters."""
        # A noninformative Gaussian prior is used for regularization
        param = [getattr(self, f.name) for f in jdc.fields(self.Param)]
        param_vec = jax.flatten_util.ravel_pytree(param)[0]
        param_prior = jsp.stats.norm.logpdf(param_vec, scale=10).sum()
        x0_prior = jsp.stats.norm.logpdf(x0, scale=10).sum()
        return x0_prior + param_prior


if __name__ == "__main__":
    args = tyro.cli(CLIArguments, description=__doc__)

    # Load Datafiles (all segments)
    d2r = np.pi / 180
    rawdata = [np.loadtxt(f.expanduser()) for f in args.datafiles]
    data = [None] * len(rawdata)
    for i, rawseg in enumerate(rawdata):
        y = jnp.c_[rawseg[:, 12] * d2r, rawseg[:, 7] * d2r, rawseg[:, 3]]
        u = jnp.c_[rawseg[:, 21] * d2r]
        data[i] = sem.Data(y, u)
    dataest = data[:-1]
    dataval = data[-1]

    # Create the PRNG keys
    key = jax.random.key(0)
    key, init_key = jax.random.split(key)

    # Create the model, estimator, and initial guess
    model = DimShortPeriod()
    est = sem.Estimator(model)
    param = est.param(dataest[0], init_key)

    # Flatten initial guess pytree
    paramvec, unpack = jax.flatten_util.ravel_pytree(param)

    # Create optimization functions
    cost = jax.jit(lambda v: -est.elbo(unpack(v), dataest[0]))
    grad = jax.jit(jax.grad(cost))
    hess_ind = est.elbo_hessian(unpack(paramvec), dataest[0])[1]
    hess_tril = hess_ind[0] >= hess_ind[1]
    hess_val = jax.jit(
        lambda v: -est.elbo_hessian(unpack(v), dataest[0])[0][hess_tril]
    )

    # Optimize
    class Obj:
        objective = cost
        gradient = grad
        hessianstructure = lambda: [i[hess_tril] for i in hess_ind]
        hessian = lambda x, lamb, objf: objf * hess_val(x)

    nlp = cyipopt.Problem(
        n=len(paramvec),
        m=0,
        problem_obj=Obj,
    )
    nlp.add_option("linear_solver", "ma57")
    paramvecopt, info = nlp.solve(paramvec)

    # Unpack optimal solution vector into pytree
    paramopt = unpack(jnp.astype(paramvecopt, paramvec.dtype))
    popt = paramopt.p

    # Obtain the optimum model and estimator outputs
    mdlopt = model.bind(paramopt.p)
    yopt = mdlopt.h(paramopt.mu, dataest[0].u)
    fopt = mdlopt.fc(paramopt.mu, dataest[0].u)
    xdotopt = jnp.diff(paramopt.mu, axis=0) / model.dt
    t = np.arange(len(dataest[0])) * model.dt

    # Plot results on screen
    if args.plot:
        from matplotlib import pyplot as plt

        for j in range(model.ny):
            plt.figure()
            plt.plot(t, dataest[0].y[:, j], ".")
            plt.plot(t, yopt[:, j], "-")
            plt.xlabel("Time [s]")
            plt.ylabel(f"Output {j}")
            plt.title("Estimation Outputs")
        for j in range(model.nx):
            plt.figure()
            plt.plot(t[1:], xdotopt[:, j], ".")
            plt.plot(t[1:], fopt[1:, j], "-")
            plt.xlabel("Time [s]")
            plt.ylabel(f"xdot {j}")
            plt.title("Estimated state-path derivative")
