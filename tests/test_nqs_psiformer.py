"""Validate the continuum spinor Psiformer ansatz.

  (1) ZERO-INIT BACKFLOW == exact Slater trial: logPsi matches slater_reference, and the
      validated local energy gives E_loc = E_gs to machine precision. (The Psiformer starts
      exactly at the warm-start non-interacting GS.)
  (2) ANTISYMMETRY: exchanging two electrons (r_i,s_i)<->(r_j,s_j) flips the sign of Psi.
  (3) BACKFLOW ON: with nonzero backflow params logPsi moves and E_loc stays finite (no NaN).

Run:  python tests/test_nqs_psiformer.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from src.nqs.rashba_reference import (noninteracting_gs, occupied_orbital_matrix,
                                      planewave_envelope)
from src.nqs.slater_reference import make_logpsi_slater
from src.nqs.local_energy import make_local_energy
from src.nqs.psiformer import ContinuumSpinorPsiformer, make_logpsi


def _build(N=5, L=7.0, lr=1.0, hz=0.5, seed=0, **mkw):
    g = noninteracting_gs(N=N, L=L, m_star=1.0, lambda_r=lr, h_z=hz, n_cut=10, twist=(0.1, 0.17))
    Kvecs, Chi = occupied_orbital_matrix(g)
    Gs, W = planewave_envelope(Kvecs, Chi)
    model = ContinuumSpinorPsiformer(n_elec=N, Gs=Gs, W=W, L=L,
                                     embed_dim=16, depth=2, n_heads=4, n_freq=4, **mkw)
    R0 = jnp.asarray(np.random.default_rng(seed).uniform(0, L, (N, 2)))
    S0 = jnp.asarray(np.random.default_rng(seed + 1).integers(0, 2, N))
    params = model.init(jax.random.PRNGKey(seed), R0, S0)
    return g, model, params, Kvecs, Chi


def test_zero_backflow_equals_slater_and_Egs():
    g, model, params, Kvecs, Chi = _build()
    logpsi = make_logpsi(model, params)
    logpsi_sl = make_logpsi_slater(Kvecs, Chi)
    rng = np.random.default_rng(3)
    for _ in range(8):
        R = jnp.asarray(rng.uniform(0, g["L"], (g["N"], 2)))
        S = jnp.asarray(rng.integers(0, 2, g["N"]))
        d = complex(logpsi(R, S)) - complex(logpsi_sl(R, S))
        # equal modulo 2*pi*i (determinant phase branch)
        assert abs(d.real) < 1e-9 and abs((d.imag + np.pi) % (2 * np.pi) - np.pi) < 1e-9, \
            f"zero-backflow Psiformer != Slater trial (dlog={d})"
    E_loc = make_local_energy(logpsi, m_star=1.0, lambda_r=1.0, h_z=0.5)
    vals = []
    for _ in range(8):
        R = jnp.asarray(rng.uniform(0, g["L"], (g["N"], 2)))
        S = jnp.asarray(rng.integers(0, 2, g["N"]))
        vals.append(complex(E_loc(R, S)))
    vals = np.array(vals)
    assert np.std(vals.real) < 1e-6, f"E_loc not zero-variance at init: std {np.std(vals.real):.2e}"
    assert abs(vals.real.mean() - g["E_gs"]) < 1e-6, \
        f"init E={vals.real.mean():.8f} != E_gs={g['E_gs']:.8f}"


def test_antisymmetry():
    g, model, params, _, _ = _build()
    logpsi = make_logpsi(model, params)
    rng = np.random.default_rng(5)
    R = jnp.asarray(rng.uniform(0, g["L"], (g["N"], 2)))
    S = jnp.asarray(rng.integers(0, 2, g["N"]))
    L0 = complex(logpsi(R, S))
    perm = jnp.arange(g["N"]).at[0].set(1).at[1].set(0)
    L1 = complex(logpsi(R[perm], S[perm]))
    ratio = np.exp(L1 - L0)
    assert abs(ratio + 1.0) < 1e-9, f"exchange did not flip sign: Psi'/Psi = {ratio}"


def test_backflow_on_is_finite_and_moves():
    g, model, params, _, _ = _build(seed=7)
    # inject small random backflow kernel
    flat = jax.tree_util.tree_map(lambda x: x, params)
    key = jax.random.PRNGKey(11)
    def perturb(p):
        leaves, tree = jax.tree_util.tree_flatten(p)
        keys = jax.random.split(key, len(leaves))
        leaves = [l + 1e-2 * (jax.random.normal(k, l.shape) +
                              (1j * jax.random.normal(jax.random.fold_in(k, 1), l.shape)
                               if jnp.iscomplexobj(l) else 0.0)).astype(l.dtype)
                  for l, k in zip(leaves, keys)]
        return jax.tree_util.tree_unflatten(tree, leaves)
    p2 = perturb(params)
    logpsi0 = make_logpsi(model, params)
    logpsi1 = make_logpsi(model, p2)
    E_loc1 = make_local_energy(logpsi1, m_star=1.0, lambda_r=1.0, h_z=0.5)
    rng = np.random.default_rng(9)
    moved, finite = 0, 0
    for _ in range(6):
        R = jnp.asarray(rng.uniform(0, g["L"], (g["N"], 2)))
        S = jnp.asarray(rng.integers(0, 2, g["N"]))
        if abs(complex(logpsi1(R, S)) - complex(logpsi0(R, S))) > 1e-6:
            moved += 1
        if np.isfinite(complex(E_loc1(R, S)).real):
            finite += 1
    assert moved >= 5, "backflow params did not change the wavefunction"
    assert finite == 6, "E_loc produced non-finite values with backflow on"


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
    g, model, params, Kvecs, Chi = _build()
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Psiformer: N={g['N']} embed=16 depth=2 heads=4  ->  {n_params} params; "
          f"init E_loc == E_gs == {g['E_gs']:+.6f}")
