#!/usr/bin/env python3

"""VIRTTAC-Castor short-period motion, Variational System Identification.

Data available at
    https://huggingface.co/datasets/dimasaad/VIRTTAC-Castor-sysid/
"""

import inspect
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy as jsp
import jax_dataclasses as jdc
import numpy as np
import scipy.io
import tyro

import sidpax.tree

# Use Ipopt if it is available, or fallback to scipy
try:
    from cyipopt import minimize_ipopt as minimize
except ImportError:
    from scipy.optimize import minimize

from sidpax import cli, common, mat, sem, sparse
from sidpax.modeling import (
    EulerDiscretization,
    MVNTransition,
    NormalMeasurements,
)


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
            data_dir / "TestPoint_241.mat",
            data_dir / "TestPoint_242.mat",
            data_dir / "TestPoint_243.mat",
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


@dataclass
class DimShortPeriod(NormalMeasurements, MVNTransition, EulerDiscretization):
    """Dimensional short-period motion model."""

    nx: int = 2
    """Number of states."""

    nu: int = 1
    """Number of exogenous inputs."""

    ny: int = 7
    """Number of outputs."""

    dt: float = 0.01
    """Sampling period."""

    @jdc.pytree_dataclass
    class Param:
        Q: mat.PositiveDefiniteMatrix
        y_log_std: jnp.array
        y_bias: jnp.array
        Za: float = 0.0
        Zq: float = 0.0
        Zde: float = 0.0
        Ma: float = 0.0
        Mq: float = 0.0
        Mde: float = 0.0
        V: float = 0.0

    @classmethod
    def param(cls, data=None, rng=None):
        """Initialize the parameter structure."""
        Q = mat.LExpDLT.identity(cls.nx)
        y_log_std = jnp.zeros(cls.ny)
        y_bias = jnp.zeros(cls.ny)
        return cls.Param(Q=Q, y_log_std=y_log_std, y_bias=y_bias)

    @common.jax_vectorize_method(signature="(x),(u)->(x)")
    def fc(self, x, u):
        """Drift function."""
        # Unpack arguments
        alpha, q = x
        (dele,) = u

        # Unpack model parameters
        Za = self.Za
        Zq = self.Zq
        Zde = self.Zde
        Ma = self.Ma
        Mq = self.Mq
        Mde = self.Mde

        # Compute state derivatives
        alphadot = Za * alpha + (Zq + 1) * q + Zde * dele
        qdot = Ma * alpha + Mq * q + Mde * dele

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
        Za = self.Za
        Zq = self.Zq
        Zde = self.Zde
        V = self.V

        # Compute alphadot
        alphadot = Za * alpha + (Zq + 1) * q + Zde * dele
        az = V * (alphadot - q)

        # Assemble output vector and return
        return jnp.array([alpha] * 4 + [q] * 3) + self.y_bias

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
    g0 = 9.80665
    data = [None] * len(args.datafiles)
    for i, f in enumerate(args.datafiles):
        matfile = scipy.io.loadmat(f.expanduser())
        out = matfile["VIRTTAC_SimData"]["Outputs"][0, 0][0, 0]
        dt = jnp.diff(out["Time"].flatten()[:2])[0]
        de_left = out["Elevator_LH_deg"] * d2r
        de_right = out["Elevator_RH_deg"] * d2r
        alpha = [out[f"alpha_ADSP{i+1}_deg"] * d2r for i in range(4)]
        q = [out[f"q_IRU{i+1}_deg_per_s"] * d2r for i in range(3)]
        az = [out[f"az_IRU{i+1}_g"] * d2r for i in range(2)]
        for alpha_i in alpha:
            alpha_i[1::3] = np.nan
            alpha_i[2::3] = np.nan
        for q_i in q:
            q_i[1::2] = np.nan
        for az_i in az:
            az_i[1::2] = np.nan
        y = jnp.c_[*alpha, *q]
        u = jnp.c_[(de_left + de_right) / 2]
        data[i] = sem.Data(y, u)
    dataest = data[:-1]
    dataval = data[-1]

    # Create the PRNG keys
    key = jax.random.key(0)
    key, init_key = jax.random.split(key)

    # Create the model and estimator classes
    model = DimShortPeriod(dt=dt)
    est = sem.Estimator(model)

    # Merge the parameters of the datasets
    params = [est.param(d, init_key) for d in dataest]
    is_unique = sem.Estimator.Param(
        p=True, mu=False, Sigma_cond=False, S_cross=False
    )
    merged = sem.merge_trees(is_unique, *params)
    merged_ind = sparse.pytree_ind(merged)

    # Pack the estimation parameters into a single vector
    paramvec, unpack = jax.flatten_util.ravel_pytree(merged)
    nparam = len(paramvec)

    # Create the problem bounds
    paramvec_low = -jnp.ones_like(paramvec) * jnp.inf
    paramvec_high = jnp.ones_like(paramvec) * jnp.inf
    bounds = jnp.c_[paramvec_low, paramvec_high]

    @jax.jit
    def cost(paramvec):
        merged = unpack(paramvec)
        return sum(-est.elbo(p, d) for p, d in zip(merged, dataest))

    @jax.jit
    def elbo_hess(paramvec):
        merged = unpack(paramvec)
        coo = [
            est.elbo_hessian(p, d, i)
            for p, d, i in zip(merged, dataest, merged_ind)
        ]
        return sparse.concatenate_coo(*coo)

    grad = jax.jit(jax.grad(cost))
    hess = lambda v: -sparse.coo_array(elbo_hess(v))

    options = dict(maxiter=args.maxiter)
    if "ipopt" in inspect.getmodule(minimize).__name__:
        method = None
        options["disp"] = 5
        options["linear_solver"] = "ma57"
    else:
        method = "trust-constr"
        options["verbose"] = 2

    result = minimize(
        cost,
        paramvec,
        bounds=bounds,
        method=method,
        jac=grad,
        hess=hess,
        options=options,
    )

    # Unpack optimization result into pytrees
    mergedopt = unpack(jnp.astype(result.x, paramvec.dtype))
    popt = mergedopt[0].p

    # Obtain the optimum model and estimator outputs
    mdlopt = model.bind(popt)
    y = np.concatenate([d.y for d in dataest])
    u = np.concatenate([d.u for d in dataest])
    mu = np.concatenate([p.mu for p in mergedopt])
    yopt = mdlopt.h(mu, u)
    fopt = mdlopt.fc(mu, u)
    xdotopt = jnp.diff(mu, axis=0) / model.dt
    freesim = [
        mdlopt.free_sim(p.mu[0], d.u) for p, d in zip(mergedopt, dataest)
    ]
    xsim = np.concatenate([pair[0] for pair in freesim])
    ysim = np.concatenate([pair[1] for pair in freesim])
    xdotsim = mdlopt.fc(xsim, u)
    t = np.arange(len(u)) * model.dt

    # Plot results on screen
    if args.plot:
        from matplotlib import pyplot as plt

        for j in range(model.ny):
            plt.figure()
            plt.plot(t, y[:, j], ".")
            plt.plot(t, yopt[:, j], "-")
            plt.plot(t, ysim[:, j], ":")
            plt.xlabel("Time [s]")
            plt.ylabel(f"Output {j}")
            plt.title(f"Estimation output {j}")
        for j in range(model.nx):
            plt.figure()
            plt.plot(t[1:], xdotopt[:, j], ".")
            plt.plot(t[1:], fopt[1:, j], "-")
            plt.plot(t, xdotsim[:, j], ":")
            plt.xlabel("Time [s]")
            plt.ylabel(f"xdot {j}")
            plt.title(f"Derivative of state {j} path")
