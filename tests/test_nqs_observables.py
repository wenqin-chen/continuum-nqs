"""Validate the spin-texture / skyrmion-number / structure-factor observable.

Key analytic check: a SPIN-INDEPENDENT trial Psi(R,S)=phi(R) is the uniform spin superposition
= every electron in the +x eigenstate, so <sigma> = (1,0,0) EXACTLY. The estimator must reproduce
m_x=1, m_y=0 to machine precision (sigma_x ratio = 1, sigma_y real part = 0), and m_z ~ 0 by sampling.
Berg-Luscher is cross-checked against the established pinning.berg_luscher_realspace on a skyrmion field.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import numpy as np
import jax.numpy as jnp

from src.nqs.observables import (spin_texture, berg_luscher, structure_factor, classify_state)
from src.nqs.pinning import make_skx_pin, berg_luscher_realspace


def test_spin_independent_trial_is_x_polarized():
    """Psi(R,S)=phi(R): exact +x polarization -> m_x=1, m_y=0 (machine precision), m_z~0."""
    def logpsi(R, S):
        return (-0.1 * jnp.sum(R ** 2)).astype(jnp.complex128)   # spin-independent
    rng = np.random.default_rng(0)
    N, B, L = 8, 1500, 5.0
    R = rng.uniform(0, L, (B, N, 2))
    S = rng.integers(0, 2, (B, N))
    A = np.array([[L, 0.0], [0.0, L]])
    tex = spin_texture(logpsi, R, S, A, ngrid=10)
    mbar = tex["mbar"]
    assert abs(mbar[0] - 1.0) < 1e-9, f"m_x should be exactly 1, got {mbar[0]}"
    assert abs(mbar[1]) < 1e-9, f"m_y should be exactly 0, got {mbar[1]}"
    assert abs(mbar[2]) < 0.12, f"m_z should be ~0 (random S), got {mbar[2]}"
    # uniform +x texture has no winding
    assert abs(berg_luscher(tex["m"])) < 0.15
    # in-plane structure factor peaks at Gamma (q=0)
    sf = structure_factor(logpsi, R, S, A, nshell=3)
    qn = np.linalg.norm(sf["q"], axis=1)
    g = sf["Sperp"][qn < 1e-9][0]
    assert g > 3.0 * sf["Sperp"][qn > 1e-9].max(), "Sperp must peak at Gamma for uniform in-plane"
    cl = classify_state(logpsi, R, S, A, ngrid=10, nshell=3)
    assert "in-plane" in cl["label"] or "Rashba" in cl["label"], cl["label"]


def test_berg_luscher_uniform_is_zero():
    m = np.zeros((16, 16, 3)); m[..., 2] = 1.0          # all spins up
    assert abs(berg_luscher(m)) < 1e-9


def test_berg_luscher_matches_reference_on_skyrmion():
    """My berg_luscher(m_grid) == pinning.berg_luscher_realspace(field, A) on a triple-Q SkX field,
    and the value is a nonzero integer (a real skyrmion crystal)."""
    qstar = 0.9548
    q1 = qstar * np.array([1.0, 0.0])
    q2 = qstar * np.array([-0.5, np.sqrt(3) / 2])
    A = 2.0 * np.pi * np.linalg.inv(np.array([q1, q2]))         # one magnetic cell
    field = make_skx_pin(q1, q2, h_pin=1.0, m0=0.5)
    ng = 30
    fr = (np.arange(ng) + 0.5) / ng
    m = np.zeros((ng, ng, 3))
    for i, u in enumerate(fr):
        for j, v in enumerate(fr):
            r = A @ np.array([u, v])
            m[i, j] = np.array(field(jnp.asarray(r)))
    Q_mine = berg_luscher(m)
    Q_ref = berg_luscher_realspace(field, A, ngrid=ng)
    assert abs(Q_mine - Q_ref) < 1e-6, f"{Q_mine} vs ref {Q_ref}"
    assert abs(Q_mine) > 0.5, f"skyrmion field must have |Q|>0, got {Q_mine}"
    assert abs(Q_mine - round(Q_mine)) < 0.05, f"Q should be near-integer, got {Q_mine}"


def test_order_parameter_uniform_has_no_finite_G():
    """Spin-independent (uniform +x) trial, UNIFORM density over the magnetic supercell (arms commensurate
    with the cell, as in the real run): m_x(0)=1 exactly, and m(G_arm)~0 (no broken-symmetry order)."""
    from src.nqs.observables import spin_order_parameter, skx_arms
    def logpsi(R, S):
        return (0.0 * jnp.sum(R)).astype(jnp.complex128)          # flat -> uniform density
    q1 = 0.9548 * np.array([1.0, 0.0]); q2 = 0.9548 * np.array([-0.5, np.sqrt(3) / 2])
    A = 2.0 * np.pi * np.linalg.inv(np.array([q1, q2])) @ np.array([[2, 0], [0, 2]])   # arms commensurate
    rng = np.random.default_rng(2)
    N, B = 8, 3000
    R = rng.uniform(0, 1, (B, N, 2)) @ np.asarray(A).T           # uniform over the cell
    S = rng.integers(0, 2, (B, N))
    G = skx_arms(q1, q2, nharm=1)
    mG = spin_order_parameter(logpsi, R, S, A, G)               # (3, nG)
    assert abs(mG[0, 0] - 1.0) < 1e-9, "m_x(0) must be exactly 1"
    arm_amp = np.mean(np.linalg.norm(mG[:, 1:], axis=0))
    assert arm_amp < 0.15, f"uniform state at commensurate arms must have ~0 order, got {arm_amp}"


def test_structure_factor_random_spins_flat_Szz():
    """Uncorrelated spins -> S_zz(q) ~ 1 everywhere with NO finite-q Bragg peak."""
    def logpsi(R, S):
        return (-0.1 * jnp.sum(R ** 2)).astype(jnp.complex128)
    rng = np.random.default_rng(1)
    N, B, L = 8, 1200, 5.0
    R = rng.uniform(0, L, (B, N, 2)); S = rng.integers(0, 2, (B, N))
    A = np.array([[L, 0.0], [0.0, L]])
    sf = structure_factor(logpsi, R, S, A, nshell=3)
    qn = np.linalg.norm(sf["q"], axis=1)
    Szz_fin = sf["Szz"][qn > 1e-9]
    assert Szz_fin.max() < 2.5, "random spins: no z Bragg peak"
    assert 0.5 < Szz_fin.mean() < 1.6, f"S_zz ~ 1 for uncorrelated spins, got {Szz_fin.mean()}"
