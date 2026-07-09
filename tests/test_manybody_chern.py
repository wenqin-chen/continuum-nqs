"""Certification of the many-body-Chern machinery (src/nqs/manybody_chern.py) BEFORE production:
1. the MC overlap estimator == the EXACT determinant overlap det(Wa^dag Wb) for plane-wave Slater pairs;
2. the FHS assembly reproduces the KNOWN Chern number of the QWZ model (C=-1 topological / 0 trivial)
   from exact spinor overlaps on the twist grid, and is gauge-invariant under random per-point phases.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from src.nqs.manybody_chern import mc_overlap, fhs_chern
from src.nqs.sampler import run_chain
from src.nqs.cell import square_cell


def _slater_logpsi(Gs, W, A):
    """log det of spinless plane-wave orbitals phi_a(r) = sum_G W[a,G] e^{iG.r} (unit-cell normalized)."""
    Gs = jnp.asarray(Gs, float); W = jnp.asarray(W, complex); Aj = jnp.asarray(A, float)

    def logpsi(R, S):
        ph = jnp.exp(1j * (R @ Gs.T))                       # (N, nG)
        M = ph @ W.T                                        # (N, nA)
        sign, ld = jnp.linalg.slogdet(M.T)
        return ld + jnp.log(sign)
    return logpsi


def test_mc_overlap_matches_exact_determinant():
    """Two N=3 plane-wave determinants sharing an orthonormal PW basis: <det_a|det_b> = det(Wa^dag Wb)
    (per unit normalization; the normalized overlap is exactly that for unitary-orthonormal rows)."""
    rng = np.random.default_rng(7)
    L = 6.0; A = square_cell(L)
    B = 2 * np.pi / L
    Gs = np.array([[i * B, j * B] for i in range(-2, 3) for j in range(-2, 3)])  # 25 orthogonal PWs
    N, nG = 3, len(Gs)

    def rand_orbitals():
        M = rng.normal(size=(N, nG)) + 1j * rng.normal(size=(N, nG))
        q, _ = np.linalg.qr(M.conj().T)                     # orthonormal rows in the PW basis
        return q.conj().T[:N]
    Wa, Wb = rand_orbitals(), rand_orbitals()
    exact = np.linalg.det(Wa.conj() @ Wb.T)                 # <a|b> with <a|a>=<b|b>=1

    la, lb = _slater_logpsi(Gs, Wa, A), _slater_logpsi(Gs, Wb, A)
    R, S, _, _ = run_chain(la, A, N, jax.random.PRNGKey(0), n_walkers=3072, n_sweeps=40, burn=200, step=0.8)
    O, absO, ess = mc_overlap(la, lb, R, S)
    assert ess > 0.05, f"degenerate importance weights (ess={ess})"
    # MC tolerance: |O| ~ 0.1-0.6 typically; few-percent agreement expected at 3072 walkers
    assert abs(O - exact) < 0.08 * max(0.2, abs(exact)) + 0.03, (O, exact)


def _qwz_u(kx, ky, mgap):
    """Lower-band spinor of the QWZ model d = (sin kx, sin ky, mgap + cos kx + cos ky)."""
    d = np.array([np.sin(kx), np.sin(ky), mgap + np.cos(kx) + np.cos(ky)])
    nrm = np.linalg.norm(d)
    u = np.array([d[0] - 1j * d[1], -(d[2] + nrm)], complex)
    n = np.linalg.norm(u)
    if n < 1e-12:                                           # south pole: pick the other gauge
        u = np.array([-(d[2] - nrm), d[0] + 1j * d[1]], complex); n = np.linalg.norm(u)
    return u / n


@pytest.mark.parametrize("mgap,C_expect", [(-1.0, -1), (-3.0, 0)])
def test_fhs_on_qwz_known_chern(mgap, C_expect):
    nt = 12
    th = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    us = [[_qwz_u(th[i], th[j], mgap) for j in range(nt)] for i in range(nt)]
    lx = np.zeros((nt, nt), complex); ly = np.zeros((nt, nt), complex)
    for i in range(nt):
        for j in range(nt):
            ox = np.vdot(us[i][j], us[(i + 1) % nt][j]); lx[i, j] = ox / abs(ox)
            oy = np.vdot(us[i][j], us[i][(j + 1) % nt]); ly[i, j] = oy / abs(oy)
    C, F = fhs_chern(lx, ly)
    assert abs(C - C_expect) < 1e-9, (C, C_expect)
    # gauge invariance: random per-point U(1) phases leave C exactly unchanged
    rng = np.random.default_rng(3)
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, (nt, nt)))
    lx2 = np.array([[lx[i, j] * np.conj(ph[i, j]) * ph[(i + 1) % nt, j] for j in range(nt)] for i in range(nt)])
    ly2 = np.array([[ly[i, j] * np.conj(ph[i, j]) * ph[i, (j + 1) % nt] for j in range(nt)] for i in range(nt)])
    C2, _ = fhs_chern(lx2, ly2)
    assert abs(C2 - C) < 1e-9
