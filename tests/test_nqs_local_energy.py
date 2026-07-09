"""Certify the kinetic+SOC+Zeeman local energy against the EXACT non-interacting eigenstate.

For a Slater determinant of occupied Rashba eigen-spin-orbitals, H Psi = (sum_a eps_a) Psi,
so E_loc(R,S) must equal E_gs at EVERY (R,S) -- a zero-variance certificate of the local
energy (the SOC term especially). If this passes, the hardest part of the NQS machinery is
correct before any neural network is introduced.

Run:  python tests/test_nqs_local_energy.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from src.nqs.rashba_reference import noninteracting_gs, occupied_orbital_matrix
from src.nqs.slater_reference import make_logpsi_slater
from src.nqs.local_energy import make_local_energy


def _zero_variance(N, L, lr, hz, n_samples=24, seed=0):
    g = noninteracting_gs(N=N, L=L, m_star=1.0, lambda_r=lr, h_z=hz, n_cut=10,
                          twist=(0.1, 0.17))
    Kvecs, Chi = occupied_orbital_matrix(g)
    logpsi = make_logpsi_slater(Kvecs, Chi)
    E_loc = make_local_energy(logpsi, m_star=1.0, lambda_r=lr, h_z=hz)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_samples):
        R = jnp.asarray(rng.uniform(0.0, L, size=(N, 2)))
        S = jnp.asarray(rng.integers(0, 2, size=N))
        vals.append(complex(E_loc(R, S)))
    vals = np.array(vals)
    return g["E_gs"], vals


def test_zero_variance_equals_Egs():
    """E_loc == E_gs (real), Im(E_loc) ~ 0, variance ~ 0, for random R,S."""
    for (N, L, lr, hz) in [(5, 7.0, 1.0, 0.5), (7, 8.0, 1.0, 0.5), (9, 9.0, 1.0, 0.0)]:
        E_gs, vals = _zero_variance(N, L, lr, hz)
        re, im = vals.real, vals.imag
        assert np.std(re) < 1e-7, f"N={N}: E_loc has nonzero variance {np.std(re):.2e}"
        assert abs(np.mean(re) - E_gs) < 1e-7, \
            f"N={N}: mean E_loc {np.mean(re):.8f} != E_gs {E_gs:.8f}"
        assert np.max(np.abs(im)) < 1e-7, f"N={N}: Im(E_loc) not ~0 ({np.max(np.abs(im)):.2e})"


def test_zeeman_only_polarized_limit():
    """Sanity: with lambda_R=0 and large h_z, lower band is fully spin-up (s=0); the kinetic
    sum equals sum of plane-wave k^2/2 and Zeeman = -h_z/2 * N."""
    N, L, hz = 5, 9.0, 4.0
    g = noninteracting_gs(N=N, L=L, m_star=1.0, lambda_r=0.0, h_z=hz, n_cut=10,
                          twist=(0.1, 0.17))
    # all occupied should be band 0 (spin up) since h_z gap = h_z >> kinetic spacing here
    assert all(d["band"] == 0 for d in g["occ"])
    Kvecs, Chi = occupied_orbital_matrix(g)
    logpsi = make_logpsi_slater(Kvecs, Chi)
    E_loc = make_local_energy(logpsi, m_star=1.0, lambda_r=0.0, h_z=hz)
    rng = np.random.default_rng(1)
    R = jnp.asarray(rng.uniform(0.0, L, size=(N, 2)))
    S = jnp.zeros(N, dtype=int)                       # all spin-up
    E = complex(E_loc(R, S))
    Ekin = float(np.sum(np.sum(Kvecs ** 2, axis=1) / 2.0))
    assert abs(E.real - (Ekin - 0.5 * hz * N)) < 1e-7
    assert abs(E.real - g["E_gs"]) < 1e-7


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    npass = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            npass += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{npass}/{len(fns)} passed")
    # show the zero-variance certificate explicitly
    for (N, L, lr, hz) in [(5, 7.0, 1.0, 0.5), (9, 9.0, 1.0, 0.0)]:
        E_gs, vals = _zero_variance(N, L, lr, hz, n_samples=12)
        print(f"  N={N} L={L} lr={lr} hz={hz}: E_gs={E_gs:+.8f}  "
              f"mean E_loc={vals.real.mean():+.8f}  std={vals.real.std():.2e}  "
              f"max|Im|={np.abs(vals.imag).max():.2e}")
