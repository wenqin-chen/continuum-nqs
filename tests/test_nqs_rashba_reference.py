"""Validate the analytic Rashba-2DEG reference oracle (exact ground truth).

Run:  python tests/test_nqs_rashba_reference.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import numpy as np
from src.nqs.rashba_reference import (
    h0_k, bands_k, bands_analytic, noninteracting_gs, occupied_orbital_matrix,
    ring_minimum, SX, SY, SZ,
)


def test_spectrum_matches_analytic():
    """eigh of h0(k) reproduces the closed-form E_pm(k) at random k and params."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        kx, ky = rng.normal(size=2) * 2.0
        ms, lr, hz = rng.uniform(0.3, 2.0), rng.uniform(0.0, 3.0), rng.uniform(0.0, 1.0)
        El, Eu, _, _ = bands_k(kx, ky, ms, lr, hz)
        aEl, aEu = bands_analytic(kx, ky, ms, lr, hz)
        assert abs(El - aEl) < 1e-10 and abs(Eu - aEu) < 1e-10


def test_hamiltonian_is_hermitian_and_soc_convention():
    """h0 Hermitian; and the Rashba block is exactly lambda_R(sigma_x k_y + sigma_y k_x)."""
    kx, ky, lr = 0.7, -1.3, 1.1
    H = h0_k(kx, ky, m_star=1.0, lambda_r=lr, h_z=0.4)
    assert np.allclose(H, H.conj().T)
    soc = H - ((kx * kx + ky * ky) / 2.0) * np.eye(2) + 0.5 * 0.4 * SZ
    assert np.allclose(soc, lr * (SX * ky + SY * kx))


def test_ring_minimum_hz0():
    """h_z=0 lower band ring: k_R = lambda_R m*, E_min = -lambda_R^2 m*/2."""
    for ms, lr in [(1.0, 1.0), (0.5, 2.0), (1.0, 0.8)]:
        kR, Emin = ring_minimum(ms, lr, 0.0)
        assert abs(kR - lr * ms) < 1e-9
        assert abs(Emin - (-0.5 * lr ** 2 * ms)) < 1e-9


def test_gs_is_sum_of_lowest_levels_and_closed_shell():
    """GS energy is the sum of the N lowest spin-orbital levels; chosen cell is gapped
    and basis-converged (cutoff_ok)."""
    g = noninteracting_gs(N=7, L=8.0, m_star=1.0, lambda_r=1.0, h_z=0.5,
                          n_cut=10, twist=(0.1, 0.17))
    assert g["gap"] > 1e-6, "expected a closed shell (unique single-determinant GS)"
    assert g["cutoff_ok"], "highest occupied k too close to the n_cut edge"
    # brute-force the same sum independently
    from src.nqs.rashba_reference import single_particle_levels
    lv = single_particle_levels(8.0, 1.0, 1.0, 0.5, 10, (0.1, 0.17))
    Es = sorted(d["E"] for d in lv)
    assert abs(g["E_gs"] - sum(Es[:7])) < 1e-12


def test_occupied_orbitals_are_normalized_eigenstates():
    """Each warm-start spinor is unit-norm and an eigenvector of its own h0(k)."""
    g = noninteracting_gs(N=9, L=9.0, m_star=1.0, lambda_r=1.0, h_z=0.5,
                          n_cut=10, twist=(0.07, 0.13))
    Kvecs, Chi = occupied_orbital_matrix(g)
    assert Kvecs.shape == (9, 2) and Chi.shape == (9, 2)
    for a in range(9):
        chi = Chi[a]
        assert abs(np.vdot(chi, chi) - 1.0) < 1e-10
        H = h0_k(Kvecs[a, 0], Kvecs[a, 1], 1.0, 1.0, 0.5)
        Hchi = H @ chi
        lam = np.vdot(chi, Hchi)
        assert np.allclose(Hchi, lam * chi, atol=1e-9), "not an eigenstate"


def test_cutoff_convergence():
    """E_gs is stable as n_cut grows (the low-energy states are captured)."""
    kw = dict(N=11, L=10.0, m_star=1.0, lambda_r=1.0, h_z=0.5, twist=(0.1, 0.2))
    e8 = noninteracting_gs(n_cut=8, **kw)["E_gs"]
    e12 = noninteracting_gs(n_cut=12, **kw)["E_gs"]
    assert abs(e8 - e12) < 1e-12, "GS energy must not depend on n_cut once converged"


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
    # human-readable reference dump for a few candidate validation cells
    print("\nCandidate G1 validation cells (non-interacting, gapped):")
    for N, L, hz in [(5, 7.0, 0.5), (7, 8.0, 0.5), (9, 9.0, 0.5), (11, 10.0, 0.5)]:
        g = noninteracting_gs(N=N, L=L, lambda_r=1.0, h_z=hz, n_cut=12, twist=(0.1, 0.17))
        print(f"  N={N:2d} L={L:4.1f} hz={hz}: E_gs={g['E_gs']:+.6f} E_per={g['E_per']:+.6f} "
              f"gap={g['gap']:.4f} n_dens={g['density']:.4f} cutoff_ok={g['cutoff_ok']}")
