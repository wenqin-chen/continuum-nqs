"""Minimal demo (CPU, ~1 min): exactness at initialization + a short training run.

1. Build the spinful Psiformer warm-started at the exact non-interacting Rashba ground
   state. Because the backflow is zero-initialized, the network IS that state: the local
   energy equals E_gs at every sampled configuration (a zero-variance certificate).
2. Perturb the backflow (pushing the energy above E_gs), then train with plain Adam:
   the variational energy descends back toward E_gs.

Run:  python examples/minimal_demo.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from src.nqs.rashba_reference import (noninteracting_gs, occupied_orbital_matrix,
                                      planewave_envelope)
from src.nqs.psiformer import ContinuumSpinorPsiformer, make_logpsi
from src.nqs.local_energy import make_local_energy
from src.nqs.sampler import run_chain, vmc_energy
from src.nqs.train import train

# --- system: N electrons in a periodic box with Rashba SOC + Zeeman field ---------------
N, L, lambda_r, h_z = 3, 6.0, 1.0, 0.5
g = noninteracting_gs(N=N, L=L, m_star=1.0, lambda_r=lambda_r, h_z=h_z,
                      n_cut=10, twist=(0.1, 0.17))
print(f"exact non-interacting ground state:  E_gs = {g['E_gs']:+.8f}")

# --- model, warm-started at the exact ground state --------------------------------------
Kvecs, Chi = occupied_orbital_matrix(g)
Gs, W = planewave_envelope(Kvecs, Chi)
model = ContinuumSpinorPsiformer(n_elec=N, Gs=Gs, W=W, L=L,
                                 embed_dim=16, depth=2, n_heads=4, n_freq=4)
rng = np.random.default_rng(0)
R0 = jnp.asarray(rng.uniform(0, L, (N, 2)))
S0 = jnp.asarray(rng.integers(0, 2, N))
params = model.init(jax.random.PRNGKey(0), R0, S0)

logpsi = make_logpsi(model, params)
E_loc = make_local_energy(logpsi, 1.0, lambda_r, h_z)
R, S, _, _ = run_chain(logpsi, L, N, jax.random.PRNGKey(7),
                       n_walkers=128, n_sweeps=60, burn=60, step=0.5)
E, Eerr, e = vmc_energy(E_loc, R, S)
print(f"network at initialization:           E    = {float(E):+.8f}"
      f"   (std of local energy: {float(jnp.std(jnp.real(e))):.2e}  <- zero-variance)")

# --- perturb the backflow, then train back with Adam ------------------------------------
def perturb_backflow(params, scale, key):
    def f(path, x):
        if any("backflow" in str(p.key) for p in path if hasattr(p, "key")):
            kr, ki = jax.random.split(jax.random.fold_in(key, hash(str(path)) % (2**31)))
            return x + scale * (jax.random.normal(kr, x.shape) +
                                1j * jax.random.normal(ki, x.shape)).astype(x.dtype)
        return x
    return jax.tree_util.tree_map_with_path(f, params)

params = perturb_backflow(params, 0.15, jax.random.PRNGKey(2))
logpsi = make_logpsi(model, params)
R, S, _, _ = run_chain(logpsi, L, N, jax.random.PRNGKey(8),
                       n_walkers=128, n_sweeps=60, burn=60, step=0.5)
E_pert, _, _ = vmc_energy(make_local_energy(logpsi, 1.0, lambda_r, h_z), R, S)
print(f"after perturbing the backflow:       E    = {float(E_pert):+.8f}")

params, hist = train(model, params, L, N, jax.random.PRNGKey(3),
                     m_star=1.0, lambda_r=lambda_r, h_z=h_z,
                     n_walkers=128, n_steps=80, lr=3e-3, n_sweeps=8,
                     step_size=0.5, burn=40, log_every=20)
print(f"after 80 Adam steps:                 E    = {hist['E'][-1]:+.8f}"
      f"   (target E_gs = {g['E_gs']:+.8f})")
