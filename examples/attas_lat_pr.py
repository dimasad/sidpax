#!/usr/bin/env python3

"""ATTAS lateral-directional motion, ny=2, Variational System Identification.

Based on data and code (test case 03) from
"Flight Vehicle System Identification - A Time Domain Methodology"
Second Edition
Author: Ravindra V. Jategaonkar
Published by AIAA, Reston, VA 20191, USA

Data available at
    https://arc.aiaa.org/doi/suppl/10.2514/4.102790/suppl_file/flt_data.zip

Code available at
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

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy as jsp
import jax_dataclasses as jdc
import numpy as np
import tyro
from scipy import optimize

from sidpax import cli, common, mat, sem, sparse
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
            data_dir / "fAttasAil1_pqrDot.asc",
            data_dir / "fAttasRud1_pqrDot.asc",
            data_dir / "fAttasAilRud2.asc",
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


class DimLatPR(MVNMeasurement, MVNTransition, EulerDiscretization):
    """Dimensional lateral-directional motion model with 2 outputs."""

    nx: int = 2
    """Number of states."""

    nu: int = 3
    """Number of exogenous inputs."""

    ny: int = 2
    """Number of outputs."""

    dt: float = 0.04
    """Sampling period."""

    @jdc.pytree_dataclass
    class Param:
        Q: mat.PositiveDefiniteMatrix
        R: mat.PositiveDefiniteMatrix
        Lp: float = 0.0
        Lr: float = 0.0
        Lda: float = 0.0
        Ldr: float = 0.0
        Lbeta: float = 0.0
        L0: float = 0.0
        Np: float = 0.0
        Nr: float = 0.0
        Nda: float = 0.0
        Ndr: float = 0.0
        Nbeta: float = 0.0
        N0: float = 0.0

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
        p, r = x
        dela, delr, beta = u

        # Unpack model parameters
        Lp = self.Lp
        Lr = self.Lr
        Lda = self.Lda
        Ldr = self.Ldr
        Lbeta = self.Lbeta
        L0 = self.L0
        Np = self.Np
        Nr = self.Nr
        Nda = self.Nda
        Ndr = self.Ndr
        Nbeta = self.Nbeta
        N0 = self.N0

        # Compute state derivatives
        pdot = Lp * p + Lr * r + Lda * dela + Ldr * delr + Lbeta * beta + L0
        rdot = Np * p + Nr * r + Nda * dela + Ndr * delr + Nbeta * beta + N0

        # Assemble state derivative vector
        xdot = jnp.array([pdot, rdot])
        return xdot

    @common.jax_vectorize_method(signature="(x),(u)->(y)")
    def h(self, x, u):
        """Output function."""
        return x

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
        y = jnp.c_[rawseg[:, 6] * d2r, rawseg[:, 8] * d2r]
        u = jnp.c_[
            (rawseg[:, 28] - rawseg[:, 27]) * d2r / 2,
            rawseg[:, 29] * d2r,
            rawseg[:, 13] * d2r,
        ]
        data[i] = sem.Data(y, u)
    dataest = data[:-1]
    dataval = data[-1]

    # Create the PRNG keys
    key = jax.random.key(0)
    key, init_key = jax.random.split(key)

    model = DimLatPR()
    est = sem.Estimator(model)
    params = [est.param(d, init_key) for d in dataest]
    is_unique = sem.Estimator.Param(
        p=True, mu=False, Sigma_cond=False, S_cross=False
    )
    merged = sem.merge_trees(is_unique, *params)
    merged_ind = sparse.pytree_ind(merged)
    paramvec, unpack = jax.flatten_util.ravel_pytree(merged)
    nparam = len(paramvec)

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

    result = optimize.minimize(
        cost,
        paramvec,
        method="trust-constr",
        jac=grad,
        hess=hess,
        options=dict(verbose=2, maxiter=args.maxiter),
    )

    mergedopt = unpack(jnp.astype(result.x, paramvec.dtype))
    popt = mergedopt[0].p

    mdlopt = model.bind(popt)
    yopt = [mdlopt.h(param.mu, d.u) for param, d in zip(mergedopt, dataest)]
