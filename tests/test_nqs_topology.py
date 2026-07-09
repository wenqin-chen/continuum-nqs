"""Validate the topology diagnostics.

Layer 1 (FHS single-particle Chern oracle):
  - QWZ lower band: |C|=1 in the topological window (0<M<2 and -2<M<0), C=0 trivial (|M|>2),
    and the sign flips under M -> -M.
  - regularized continuum Dirac cone: C=+1 (M,B>0), C=0 (M<0).
These pin the oracle that the rotation-eigenvalue many-body estimator (Layer 2) must reproduce.

Run:  python tests/test_nqs_topology.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import numpy as np
from src.nqs.topology import chern_qwz, chern_fhs, occupied_grid, dirac_h


def test_qwz_topological_window():
    c_p = chern_qwz(M=1.0, Nk=24)     # 0<M<2  -> |C|=1
    c_m = chern_qwz(M=-1.0, Nk=24)    # -2<M<0 -> opposite sign
    assert abs(abs(c_p) - 1.0) < 1e-6, f"QWZ M=1 not |C|=1: {c_p}"
    assert abs(abs(c_m) - 1.0) < 1e-6, f"QWZ M=-1 not |C|=1: {c_m}"
    assert np.sign(round(c_p)) == -np.sign(round(c_m)), "Chern did not flip sign under M->-M"


def test_qwz_trivial():
    assert abs(chern_qwz(M=3.0, Nk=24)) < 1e-6, "QWZ M=3 should be trivial C=0"
    assert abs(chern_qwz(M=-3.0, Nk=24)) < 1e-6, "QWZ M=-3 should be trivial C=0"


def test_qwz_grid_convergence():
    """Integer Chern is grid-robust (FHS is exact for any grid resolving the gap)."""
    assert abs(chern_qwz(M=1.0, Nk=16) - chern_qwz(M=1.0, Nk=32)) < 1e-6


def test_continuum_dirac_cone():
    """Regularized Dirac cone over a large k-patch: C=+1 (M>0), C=0 (M<0,B>0)."""
    Lam, Nk = 9.0, 41
    ks = np.linspace(-Lam, Lam, Nk)
    u_top = occupied_grid(lambda kx, ky: dirac_h(kx, ky, M=1.0, B=1.0), ks, ks, n_occ=1)
    u_triv = occupied_grid(lambda kx, ky: dirac_h(kx, ky, M=-1.0, B=1.0), ks, ks, n_occ=1)
    c_top = chern_fhs(u_top, periodic=True)
    c_triv = chern_fhs(u_triv, periodic=True)
    # lower-band C = -[sign(M)+sign(B)]/2 -> -1 (M,B>0), 0 (M<0,B>0). Test |C| (sign is convention).
    assert abs(abs(c_top) - 1.0) < 1e-2, f"continuum Dirac |C| != 1: {c_top}"
    assert abs(c_triv) < 1e-2, f"continuum trivial C != 0: {c_triv}"
    assert abs(c_top - c_triv) > 0.9, "topological/trivial not distinguished"


import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from src.nqs.topology import (rotation_eigenvalue_exact_singleG, rotation_eigenvalue_vmc,
                              spin_rotation_phase)
from src.nqs.slater_reference import make_logpsi_general
from src.nqs.rashba_reference import planewave_envelope

_U4 = np.array([[np.exp(-1j * np.pi / 4), 0], [0, np.exp(1j * np.pi / 4)]])  # exp(-i pi sigma_z/4)


def _c4_state(L, chi0, symmetric=True):
    """4-orbit G-set {(1,0),(0,1),(-1,0),(0,-1)}*2pi/L. If symmetric, spinors are U4^k chi0 so the
    Slater det is an exact C4 eigenstate; else arbitrary spinors (C4-closed basis, non-symmetric)."""
    g = 2 * np.pi / L
    orbit = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]  # R4 cycles
    Gs = g * np.stack(orbit).astype(float)
    if symmetric:
        Chi = np.stack([np.linalg.matrix_power(_U4, k) @ chi0 for k in range(4)])
    else:
        rng = np.random.default_rng(2)
        Chi = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
        Chi = Chi / np.linalg.norm(Chi, axis=1, keepdims=True)
    return Gs, Chi


def _logpsi_np(Gs, Chi):
    Gpw, W = planewave_envelope(Gs, Chi)
    lp = make_logpsi_general(Gpw, W)
    return lambda Ro, So: complex(lp(jnp.asarray(Ro, float), jnp.asarray(So, int)))


def test_rotation_c4_symmetric_zero_variance():
    """Exact C4 eigenstate: <R_4> estimator = det(m) with ZERO variance (ratio constant)."""
    L = 7.0
    Gs, Chi = _c4_state(L, np.array([1.0, 0.4 + 0.2j]) / np.linalg.norm([1.0, 0.4 + 0.2j]), True)
    exact = rotation_eigenvalue_exact_singleG(Gs, Chi, 4, L)
    lp = _logpsi_np(Gs, Chi)
    rng = np.random.default_rng(0)
    R = rng.uniform(0, L, (40, 4, 2))
    S = rng.integers(0, 2, (40, 4))
    vmc, err = rotation_eigenvalue_vmc(lp, R, S, 4, L)
    assert abs(exact) > 0.999, f"symmetric state should have |<R4>|=1, got {abs(exact):.4f}"
    assert err.real < 1e-7 and err.imag < 1e-7, f"not zero-variance (err {err})"
    assert abs(vmc - exact) < 1e-7, f"VMC <R4>={vmc} != exact {exact}"


def test_rotation_general_matches_exact_mcmc():
    """Non-symmetric (C4-closed basis): MCMC |psi|^2 estimator matches det(m) (|.|<1)."""
    from src.nqs.sampler import run_chain
    L = 7.0
    Gs, Chi = _c4_state(L, None, symmetric=False)
    exact = rotation_eigenvalue_exact_singleG(Gs, Chi, 4, L)
    Gpw, W = planewave_envelope(Gs, Chi)
    lp_j = make_logpsi_general(Gpw, W)
    R, S, _, _ = run_chain(lp_j, L, 4, jax.random.PRNGKey(0), n_walkers=4000,
                           n_sweeps=200, burn=150, step=0.6)
    lp = lambda Ro, So: complex(lp_j(jnp.asarray(Ro, float), jnp.asarray(So, int)))
    vmc, err = rotation_eigenvalue_vmc(lp, np.array(R), np.array(S), 4, L)
    assert abs(vmc - exact) < 5 * abs(err) + 5e-3, \
        f"MCMC <R4>={vmc:.4f} != exact {exact:.4f} (err {abs(err):.1e})"


def test_c2_consistency():
    """C_2 = (C_4)^2: <R_2> on the symmetric state equals det(m) for n=2."""
    L = 7.0
    c0 = np.array([1.0, 0.4 + 0.2j]); c0 = c0 / np.linalg.norm(c0)   # non-polarized (psi!=0)
    Gs, Chi = _c4_state(L, c0, True)
    exact2 = rotation_eigenvalue_exact_singleG(Gs, Chi, 2, L)
    lp = _logpsi_np(Gs, Chi)
    rng = np.random.default_rng(1)
    R = rng.uniform(0, L, (30, 4, 2)); S = rng.integers(0, 2, (30, 4))
    vmc2, err2 = rotation_eigenvalue_vmc(lp, R, S, 2, L)
    assert abs(vmc2 - exact2) < 1e-7, f"C2: VMC {vmc2} != exact {exact2}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    npass = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}"); npass += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{npass}/{len(fns)} passed")
    for M in (-3.0, -1.0, 1.0, 3.0):
        print(f"  QWZ M={M:+.1f}: C_FHS = {chern_qwz(M, Nk=24):+.4f}")
