"""Certify the VMC training loop against a KNOWN minimum.

The non-interacting Rashba H has variational minimum E_gs, reached by the Psiformer at zero
backflow. We PERTURB the backflow (pushing E above E_gs), then train: the energy must descend
back toward E_gs. This certifies the surrogate-loss gradient (its sign especially, across the
real backbone + complex backflow params) and the optimizer plumbing end to end.

Run:  python tests/test_nqs_train.py     (~1-2 min on CPU)
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import numpy as np
import jax, jax.numpy as jnp
import optax
jax.config.update("jax_enable_x64", True)

from src.nqs.rashba_reference import (noninteracting_gs, occupied_orbital_matrix,
                                      planewave_envelope)
from src.nqs.psiformer import ContinuumSpinorPsiformer, make_logpsi
from src.nqs.local_energy import make_local_energy
from src.nqs.sampler import run_chain, vmc_energy
from src.nqs.train import train


def _perturb_backflow(params, scale, key):
    """Add noise ONLY to the backflow params (push the trial off the exact GS)."""
    def f(path, x):
        if any("backflow" in str(p.key) for p in path if hasattr(p, "key")):
            kr, ki = jax.random.split(jax.random.fold_in(key, hash(str(path)) % (2**31)))
            return x + scale * (jax.random.normal(kr, x.shape) +
                                1j * jax.random.normal(ki, x.shape)).astype(x.dtype)
        return x
    return jax.tree_util.tree_map_with_path(f, params)


def test_training_descends_to_Egs():
    N, L, lr_soc, hz = 5, 7.0, 1.0, 0.5
    g = noninteracting_gs(N=N, L=L, m_star=1.0, lambda_r=lr_soc, h_z=hz, n_cut=10, twist=(0.1, 0.17))
    Kvecs, Chi = occupied_orbital_matrix(g)
    Gs, W = planewave_envelope(Kvecs, Chi)
    model = ContinuumSpinorPsiformer(n_elec=N, Gs=Gs, W=W, L=L,
                                     embed_dim=16, depth=2, n_heads=4, n_freq=4)
    R0 = jnp.asarray(np.random.default_rng(0).uniform(0, L, (N, 2)))
    S0 = jnp.asarray(np.random.default_rng(1).integers(0, 2, N))
    params = model.init(jax.random.PRNGKey(0), R0, S0)
    params = _perturb_backflow(params, 0.15, jax.random.PRNGKey(2))

    def energy_now(p):
        logpsi = make_logpsi(model, p)
        E_loc = make_local_energy(logpsi, 1.0, lr_soc, hz)
        R, S, _, _ = run_chain(logpsi, L, N, jax.random.PRNGKey(7),
                               n_walkers=256, n_sweeps=80, burn=80, step=0.5)
        return vmc_energy(E_loc, R, S)[0]

    E_init = energy_now(params)
    params, hist = train(model, params, L, N, jax.random.PRNGKey(3),
                         m_star=1.0, lambda_r=lr_soc, h_z=hz, n_walkers=256, n_steps=120,
                         lr=3e-3, n_sweeps=8, step_size=0.5, burn=40, log_every=30)
    E_final = energy_now(params)

    print(f"  E_gs={g['E_gs']:+.6f}  E_init={E_init:+.6f}  E_final={E_final:+.6f}")
    # Optimizer-strength-INDEPENDENT certification: training removes >=95% of the energy
    # EXCESS above the known minimum (the absolute sub-mHa residual is a production
    # MinSR/KFAC concern, PLAN sec 3a). Plus a strict variational-floor check.
    excess_init = E_init - g["E_gs"]
    frac_remaining = (E_final - g["E_gs"]) / excess_init
    print(f"  excess_init={excess_init:+.4f}  recovered={100 * (1 - frac_remaining):.1f}%")
    assert excess_init > 1e-2, "perturbation did not raise the energy above E_gs"
    assert E_final < E_init - 1e-3, f"training did not lower the energy ({E_init} -> {E_final})"
    assert frac_remaining < 0.05, \
        f"training recovered only {100 * (1 - frac_remaining):.1f}% of the excess (want >=95%)"
    assert E_final > g["E_gs"] - 0.02, "energy fell below the non-interacting GS (variational bug)"


if __name__ == "__main__":
    try:
        test_training_descends_to_Egs()
        print("\n  PASS  test_training_descends_to_Egs")
    except AssertionError as e:
        print(f"\n  FAIL  {e}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"\n  ERROR {type(e).__name__}: {e}")
