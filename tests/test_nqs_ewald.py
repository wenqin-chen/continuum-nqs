"""Validate the 2D Ewald Coulomb (the convention must match the HF: V(q)=2*pi*alpha/q).

  (1) SPLITTING INDEPENDENCE -- the load-bearing test: ewald_pair(r) and zeta_M must not depend
      on the (arbitrary) Ewald width eta once the lattice sums are converged. A wrong coefficient
      breaks this.
  (2) PERIODICITY on the torus; (3) EVENNESS V(r)=V(-r); (4) short-range V(r)->alpha/r;
  (5) linear ALPHA scaling; (6) coulomb_energy = pair sum + Madelung, translation invariant;
  (7) the full interacting local energy (kinetic+SOC+Zeeman+Coulomb) is finite for the Psiformer.

Run:  python tests/test_nqs_ewald.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from src.nqs.ewald import make_ewald


def test_splitting_independence():
    """ewald_pair(r) and zeta_M independent of eta (converged sums)."""
    L = 7.0
    r = jnp.asarray([1.3, -2.1])
    vals, zs = [], []
    for c in (0.8, 1.0, 1.25):
        ep, zM, _ = make_ewald(L, alpha=1.0, ewald_alpha=c * np.sqrt(np.pi) / L,
                               n_real=8, n_recip=14)
        vals.append(float(ep(r))); zs.append(zM)
    assert np.ptp(vals) < 1e-7, f"ewald_pair depends on eta: spread {np.ptp(vals):.2e}"
    assert np.ptp(zs) < 1e-7, f"zeta_M depends on eta: spread {np.ptp(zs):.2e}"


def test_periodicity_and_evenness():
    L = 7.0
    ep, _, _ = make_ewald(L, alpha=1.0, n_real=8, n_recip=12)
    r = jnp.asarray([1.1, 2.7])
    assert abs(float(ep(r)) - float(ep(r + jnp.array([L, 0.0])))) < 1e-8
    assert abs(float(ep(r)) - float(ep(r + jnp.array([0.0, L])))) < 1e-8
    assert abs(float(ep(r)) - float(ep(-r))) < 1e-10, "potential not even"


def test_short_range_is_alpha_over_r():
    """V(r)*|r| -> alpha as |r|->0 (the 1/r coefficient is the coupling alpha)."""
    L = 7.0
    for alpha in (1.0, 0.6):
        ep, _, _ = make_ewald(L, alpha=alpha, n_real=8, n_recip=12)
        for d in (1e-3, 1e-4):
            r = jnp.asarray([d, 0.0])
            assert abs(float(ep(r)) * d - alpha) < 1e-2 * alpha + 5e-3, \
                f"short-range coefficient off at d={d}, alpha={alpha}"


def test_alpha_scaling():
    L = 7.0
    r = jnp.asarray([1.3, -0.7])
    ep1, z1, _ = make_ewald(L, alpha=1.0, n_real=6, n_recip=12)
    ep2, z2, _ = make_ewald(L, alpha=2.5, n_real=6, n_recip=12)
    assert abs(2.5 * float(ep1(r)) - float(ep2(r))) < 1e-9
    assert abs(2.5 * z1 - z2) < 1e-9


def test_coulomb_energy_translation_invariant():
    L = 8.0
    ep, zM, ce = make_ewald(L, alpha=1.0, n_real=6, n_recip=12)
    rng = np.random.default_rng(0)
    R = jnp.asarray(rng.uniform(0, L, (6, 2)))
    E = float(ce(R))
    # explicit pair sum + Madelung
    pairs = sum(float(ep(R[i] - R[j])) for i in range(6) for j in range(i + 1, 6))
    assert abs(E - (pairs + 6 * zM / 2.0)) < 1e-9, "coulomb_energy != pair sum + Madelung"
    shift = jnp.asarray([2.3, -1.1])
    assert abs(E - float(ce(R + shift))) < 1e-8, "Coulomb energy not translation invariant"


def test_full_interacting_local_energy_finite():
    """kinetic+SOC+Zeeman+Coulomb is finite for the Psiformer at random configs."""
    from src.nqs.rashba_reference import (noninteracting_gs, occupied_orbital_matrix,
                                          planewave_envelope)
    from src.nqs.psiformer import ContinuumSpinorPsiformer, make_logpsi
    from src.nqs.local_energy import make_local_energy
    N, L, lr, hz = 5, 7.0, 1.0, 0.5
    g = noninteracting_gs(N=N, L=L, lambda_r=lr, h_z=hz, n_cut=10, twist=(0.1, 0.17))
    Kvecs, Chi = occupied_orbital_matrix(g)
    Gs, W = planewave_envelope(Kvecs, Chi)
    model = ContinuumSpinorPsiformer(n_elec=N, Gs=Gs, W=W, L=L,
                                     embed_dim=16, depth=2, n_heads=4)
    R0 = jnp.asarray(np.random.default_rng(0).uniform(0, L, (N, 2)))
    S0 = jnp.asarray(np.random.default_rng(1).integers(0, 2, N))
    params = model.init(jax.random.PRNGKey(0), R0, S0)
    logpsi = make_logpsi(model, params)
    E0 = make_local_energy(logpsi, 1.0, lr, hz)
    _, _, ce = make_ewald(L, alpha=1.0, n_real=6, n_recip=12)
    rng = np.random.default_rng(5)
    for _ in range(4):
        R = jnp.asarray(rng.uniform(0, L, (N, 2)))
        S = jnp.asarray(rng.integers(0, 2, N))
        E = complex(E0(R, S)) + float(ce(R))
        assert np.isfinite(E.real) and np.isfinite(E.imag), "non-finite interacting E_loc"


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
    ep, zM, ce = make_ewald(7.0, alpha=1.0, n_real=6, n_recip=12)
    print(f"  L=7 alpha=1: zeta_M(per elec)={zM:+.6f}; V(r=L/4)={float(ep(jnp.array([7.0/4,0.0]))):+.6f}")
