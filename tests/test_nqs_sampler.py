"""Validate the MCMC sampler against the exact Slater trial.

Two checks:
  (1) Sampled <E_loc> = E_gs. (Trivially exact since E_loc is zero-variance, but confirms the
      sampler + batched local-energy plumbing run end to end.)
  (2) The sampled one-body density is UNIFORM to MC noise -- a genuine test of detailed balance
      for the position moves (a plane-wave determinant has exactly uniform <n(r)> = N/L^2).

Run:  python tests/test_nqs_sampler.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from src.nqs.rashba_reference import noninteracting_gs, occupied_orbital_matrix
from src.nqs.slater_reference import make_logpsi_slater
from src.nqs.local_energy import make_local_energy
from src.nqs.sampler import run_chain, vmc_energy


def _setup(N=7, L=8.0, lr=1.0, hz=0.5):
    g = noninteracting_gs(N=N, L=L, m_star=1.0, lambda_r=lr, h_z=hz, n_cut=10, twist=(0.1, 0.17))
    Kvecs, Chi = occupied_orbital_matrix(g)
    logpsi = make_logpsi_slater(Kvecs, Chi)
    E_loc = make_local_energy(logpsi, m_star=1.0, lambda_r=lr, h_z=hz)
    return g, logpsi, E_loc


def test_sampled_energy_equals_Egs():
    g, logpsi, E_loc = _setup()
    R, S, ap, asp = run_chain(logpsi, g["L"], g["N"], jax.random.PRNGKey(0),
                              n_walkers=256, n_sweeps=120, burn=120, step=0.5)
    Emean, Eerr, _ = vmc_energy(E_loc, R, S)
    assert 0.1 < ap < 0.95, f"position acceptance out of range: {ap:.2f}"
    assert abs(Emean - g["E_gs"]) < 1e-6, f"<E>={Emean:.8f} != E_gs={g['E_gs']:.8f}"


def test_density_is_uniform():
    """A plane-wave determinant has exactly uniform <n(r)>=N/L^2, so binned position counts
    must fluctuate at the POISSON level, not more (clustering would mean a broken position
    sampler). Compare observed rel-std to the Poisson prediction 1/sqrt(mean_count)."""
    g, logpsi, E_loc = _setup(N=9, L=9.0)
    R, S, ap, asp = run_chain(logpsi, g["L"], g["N"], jax.random.PRNGKey(1),
                              n_walkers=2048, n_sweeps=300, burn=200, step=0.5)
    L = g["L"]
    pos = np.array(R).reshape(-1, 2) % L
    nb = 6
    H, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=nb, range=[[0, L], [0, L]])
    mean_count = H.mean()
    rel = np.std(H) / mean_count
    poisson = 1.0 / np.sqrt(mean_count)
    assert rel < 1.8 * poisson, (
        f"density fluctuates above Poisson (rel {rel:.3f} vs expected ~{poisson:.3f}); "
        f"position sampler suspect")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    npass = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}"); npass += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{npass}/{len(fns)} passed")
    g, logpsi, E_loc = _setup()
    R, S, ap, asp = run_chain(logpsi, g["L"], g["N"], jax.random.PRNGKey(2),
                              n_walkers=256, n_sweeps=120, burn=120, step=0.5)
    Emean, Eerr, _ = vmc_energy(E_loc, R, S)
    print(f"  sampled <E>={Emean:+.8f} +/- {Eerr:.1e}  (E_gs={g['E_gs']:+.8f})  "
          f"acc_pos={ap:.2f} acc_spin={asp:.2f}")
