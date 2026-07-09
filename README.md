# continuum-nqs

**Attention-based neural quantum states for interacting electrons in the continuum.**

[![tests](https://github.com/wenqin-chen/continuum-nqs/actions/workflows/tests.yml/badge.svg)](https://github.com/wenqin-chen/continuum-nqs/actions/workflows/tests.yml)

A JAX implementation of a transformer neural-network wavefunction for the two-dimensional
electron gas with Rashba spin-orbit coupling, a Zeeman field, and long-range (Ewald-summed)
Coulomb interaction. The architecture is inspired by the electron-token self-attention design
of [Psiformer](https://arxiv.org/abs/2211.13672); unlike the molecular Psiformer, the
wavefunction here is **periodic, complex-valued, and spinful**. The network is optimized by
variational Monte Carlo with natural-gradient
([MinSR](https://www.nature.com/articles/s41567-024-02566-1)) updates.

This is research code accompanying a methodology paper in preparation. Author:
[Wenqin Chen](https://wenqin-chen.github.io/).

## Architecture

- **Tokens = electrons**: periodic Fourier position features + spin one-hot, embedded and
  processed by pre-norm multi-head self-attention blocks (`psiformer.py`)
- **Complex backflow, zero-initialized**: at initialization the network is *exactly* a
  validated reference state (a plane-wave spinor Slater determinant), so training starts from
  a known-good wavefunction and learns only the correction
- **Bloch spinor envelopes**, warm-started from Hartree-Fock / non-interacting Rashba
  orbitals, optionally learnable
- **Generalized spinor determinants** (the provably correct antisymmetrization under
  spin-orbit coupling), summed over K determinants in log-space; optional RBF pair Jastrow
- **MinSR optimizer** (`minsr.py`): the natural-gradient (quantum-metric / Fisher) system
  solved in sample space through an exact push-through identity, making second-order
  optimization affordable for the full network
- **Local energy** (`local_energy.py`): kinetic + Rashba SOC + Zeeman + Ewald Coulomb, with a
  forward-Laplacian ([folx](https://github.com/microsoft/folx)) fast path
- **Metropolis sampler** over positions and spins with persistent walkers (`sampler.py`)

## Validation

The test suite (`pytest`, 60+ tests) is built around exact oracles rather than "it trains":

- **Zero-variance certificates**: on analytic eigenstates the local energy equals the exact
  eigenvalue at *every* sampled configuration, to machine precision
- **MinSR is validated against the explicit parameter-space solve** (the push-through
  identity is exact, so the two must agree to machine precision)
- **Forward-Laplacian vs dense autodiff** agreement to ~1e-16
- **Exact-diagonalization benchmark**: the trained ansatz reaches the exact ground state of
  the full interacting Hamiltonian within 0.7 mHa (96% of the correlation energy) at two
  electrons
- Non-collinear gates: exact non-interacting Rashba and spin-spiral ground states

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# exactness at initialization + a short training run (CPU, ~1 min)
python examples/minimal_demo.py

# test suite (CPU, ~4 min)
pytest tests -x -q
```

## Module map

| Module | What it does |
|---|---|
| `src/nqs/psiformer.py` | the spinful transformer wavefunction |
| `src/nqs/minsr.py` | MinSR natural-gradient optimizer |
| `src/nqs/train.py` | VMC training loop (pluggable optax optimizers) |
| `src/nqs/local_energy.py` | validated local energy (SOC, Zeeman, external fields) |
| `src/nqs/ewald.py` | 2D Ewald-summed Coulomb energy |
| `src/nqs/sampler.py` | Metropolis MCMC over positions + spins |
| `src/nqs/pinning.py` | pinning fields that nucleate competing magnetic orders |
| `src/nqs/rashba_reference.py`, `spiral_reference.py`, `slater_reference.py` | exact reference states / warm starts |
| `src/nqs/observables.py`, `topology.py`, `manybody_chern.py` | spin textures, Chern numbers, many-body topology |

## Citation

A methodology paper is in preparation. Until then, please cite this repository.

## License

MIT
