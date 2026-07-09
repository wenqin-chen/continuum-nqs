"""Validate the general 2D cell + the general-lattice Ewald.

  - cell helpers: a_i . b_j = 2*pi delta (reciprocal), frac/cart roundtrip, lattice-periodic wrap.
  - general Ewald: splitting(eta)-INDEPENDENT total energy on RECTANGULAR (aspect sqrt3) and
    HEXAGONAL/triangular cells (rectangular/hexagonal supercell shapes), linear alpha scaling, V(r)->alpha/r.
  - reduces EXACTLY to the square-box make_ewald(L) when A = L*I.

Run:  python tests/test_nqs_cell.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from src.nqs.cell import square_cell, rect_cell, hex_cell, reciprocal, cell_area, frac, wrap
from src.nqs.ewald import make_ewald, make_ewald_lattice


def test_reciprocal_orthogonality():
    for A in (square_cell(7.0), rect_cell(7.0, 7 * np.sqrt(3)), hex_cell(7.0)):
        B = reciprocal(A)
        assert np.allclose(np.asarray(A).T @ B, 2 * np.pi * np.eye(2), atol=1e-9), "a_i.b_j != 2pi delta"


def test_frac_cart_and_periodic_wrap():
    A = hex_cell(7.0)
    rng = np.random.default_rng(0)
    r = jnp.asarray(rng.uniform(-3, 10, (5, 2)))
    # wrap is lattice-periodic: wrap(r + A n) == wrap(r)
    n = jnp.asarray([[2.0, -1.0]])
    shift = (n @ jnp.asarray(A).T)
    assert np.allclose(np.array(wrap(r, A)), np.array(wrap(r + shift, A)), atol=1e-9)
    # fractional coords lie in [0,1)
    s = np.array(frac(r, A))
    assert s.min() >= 0 and s.max() < 1.0


def test_ewald_reduces_to_square():
    L = 7.0
    rng = np.random.default_rng(1)
    epL, zL, _ = make_ewald(L, alpha=1.0, n_real=4, n_recip=8)
    epA, zA, _ = make_ewald_lattice(square_cell(L), alpha=1.0, n_real=4, n_recip=8)
    assert abs(zL - zA) < 1e-12
    for _ in range(5):
        r = jnp.asarray(rng.uniform(0, L, 2))
        assert abs(float(epL(r)) - float(epA(r))) < 1e-12, "general lattice != square at A=L*I"


def _splitting_independent(A, label):
    rng = np.random.default_rng(0)
    N = 6
    Ainv = np.linalg.inv(np.asarray(A))
    R = jnp.asarray((rng.uniform(0, 1, (N, 2)) @ np.asarray(A).T))   # random points in the cell
    Es = []
    for c in (0.7, 1.0, 1.4):
        eta = c * np.sqrt(np.pi) / np.sqrt(cell_area(A))
        _, _, ce = make_ewald_lattice(A, alpha=1.0, ewald_alpha=eta, n_real=9, n_recip=16)
        Es.append(float(ce(R)))
    assert np.ptp(Es) < 1e-6, f"{label}: total energy depends on eta (spread {np.ptp(Es):.2e}): {Es}"


def test_splitting_independence_rect():
    _splitting_independent(rect_cell(7.0, 7.0 * np.sqrt(3)), "rect")


def test_splitting_independence_hex():
    _splitting_independent(hex_cell(8.0), "hex")


def test_alpha_scaling_and_short_range_general():
    A = hex_cell(8.0)
    ep1, z1, _ = make_ewald_lattice(A, alpha=1.0, n_real=8, n_recip=14)
    ep2, z2, _ = make_ewald_lattice(A, alpha=2.5, n_real=8, n_recip=14)
    r = jnp.asarray([1.3, -0.7])
    assert abs(2.5 * float(ep1(r)) - float(ep2(r))) < 1e-9 and abs(2.5 * z1 - z2) < 1e-9
    for d in (1e-3, 1e-4):                          # V(r)*|r| -> alpha as r->0
        assert abs(float(ep1(jnp.asarray([d, 0.0]))) * d - 1.0) < 2e-2


def test_sampler_and_psiformer_on_rect_cell():
    """The sampler + Psiformer run end-to-end on a RECTANGULAR cell (rectangular supercell shape):
    a plane-wave determinant with reciprocal-lattice Gs is cell-periodic, so MCMC samples a UNIFORM
    fractional-coord density, and the Psiformer evaluates + stays antisymmetric."""
    from src.nqs.slater_reference import make_logpsi_general
    from src.nqs.psiformer import ContinuumSpinorPsiformer, make_logpsi
    from src.nqs.sampler import run_chain
    A = rect_cell(7.0, 7.0 * np.sqrt(3))
    B = reciprocal(A)
    ints = [(1, 0), (0, 1), (1, 1), (2, 0)]
    Ne = len(ints)
    Gs = np.array([B @ np.array(m, float) for m in ints])
    rng = np.random.default_rng(0)
    Chi = rng.normal(size=(Ne, 2)) + 1j * rng.normal(size=(Ne, 2))
    Chi /= np.linalg.norm(Chi, axis=1, keepdims=True)
    W = np.zeros((Ne, Ne, 2), complex)
    for a in range(Ne):
        W[a, a] = Chi[a]
    lp = make_logpsi_general(Gs, W)
    R, S, ap, _ = run_chain(lp, A, Ne, jax.random.PRNGKey(0), n_walkers=2048,
                            n_sweeps=200, burn=150, step=0.6)
    pos = np.array(R).reshape(-1, 2)
    sfr = np.mod(pos @ np.linalg.inv(np.asarray(A)).T, 1.0)        # fractional coords
    Hh, _, _ = np.histogram2d(sfr[:, 0], sfr[:, 1], bins=6, range=[[0, 1], [0, 1]])
    rel, pois = np.std(Hh) / Hh.mean(), 1.0 / np.sqrt(Hh.mean())
    assert 0.1 < ap < 0.95, f"rect-cell acceptance off: {ap:.2f}"
    assert rel < 1.8 * pois, f"rect-cell density not uniform (rel {rel:.3f} vs Poisson {pois:.3f})"
    model = ContinuumSpinorPsiformer(n_elec=Ne, Gs=Gs, W=W, L=A, embed_dim=12, depth=2, n_heads=3)
    R0 = jnp.asarray(rng.uniform(0, 1, (Ne, 2)) @ np.asarray(A).T)
    S0 = jnp.asarray(rng.integers(0, 2, Ne))
    params = model.init(jax.random.PRNGKey(0), R0, S0)
    pf = make_logpsi(model, params)
    v0 = complex(pf(R0, S0))
    perm = jnp.arange(Ne).at[0].set(1).at[1].set(0)
    v1 = complex(pf(R0[perm], S0[perm]))
    assert np.isfinite(v0.real), "psiformer non-finite on rect cell"
    assert abs(np.exp(v1 - v0) + 1.0) < 1e-9, "psiformer not antisymmetric on rect cell"


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
