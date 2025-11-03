[![Testing](https://github.com/dimasad/sidpax/actions/workflows/tests.yml/badge.svg)](https://github.com/dimasad/sidpax/actions/workflows/tests.yml)

# sidpax
System Identification Programs in JAX

The main goal of this project is implementing variational smoother-error method 
for system identification, using JAX. This is described in the paper
[Parameterizations for Large-Scale Variational System Identification (arXiv:2404.10137)](https://arxiv.org/abs/2404.10137) as variational system identification. 

The functionality of the paper was first implemented in 
[dimasad/automatica-2024-code](https://github.com/dimasad/automatica-2024-code),
but as a proof-of-concept the codebase was too clunky and hard to maintain.
I rewrote the code in [dimasad/visid](https://github.com/dimasad/visid) 
using Flax Linen, but I think that was not the best approach for this. The main
reason is that SDE discretizations, Kalman filters, and other transformations
on model functions that require differentiation are clunky with Flax. It is 
more flexible, for my approach, to work with JAX directly.

I want this code to be used for system identification of aircraft as well,
similar to what was done in 
[dimasad/scitech-2025-code](https://github.com/dimasad/scitech-2025-code),
hence the name Sidpax, a homage to 
[SIDPAC](https://software.nasa.gov/software/LAR-16100-1).

```bibtex
@article{Dutra2025Parametrizations,
  title        = {Parameterizations for Large-Scale Variational System Identification Using Unconstrained Optimization},
  author       = {Dimas Abreu Archanjo Dutra},
  journal      = {Automatica},
  volume       = {173},
  pages        = {112086},
  year         = {2025},
  doi          = {10.1016/j.automatica.2024.112086}
}
```