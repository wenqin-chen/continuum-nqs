"""Validate the folx Forward-Laplacian kinetic path == the dense-Hessian reference.

The Forward Laplacian computes the SAME Laplacian (just faster, ~O(N) vs O((2N)^2)), so
make_local_energy(use_folx=True) must equal make_local_energy(use_folx=False) to machine
precision -- on the Slater, Psiformer, and spiral states, and under vmap.

Run:  python tests/test_nqs_folx.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from src.nqs.rashba_reference import noninteracting_gs, occupied_orbital_matrix, planewave_envelope
from src.nqs.slater_reference import make_logpsi_slater, make_logpsi_general
from src.nqs.local_energy import make_local_energy
from src.nqs.psiformer import ContinuumSpinorPsiformer, make_logpsi
from src.nqs.spiral_reference import spiral_gs, spiral_field


def test_folx_matches_dense_slater():
    N, L, lr, hz = 5, 7.0, 1.0, 0.5
    g = noninteracting_gs(N=N, L=L, lambda_r=lr, h_z=hz, n_cut=10, twist=(0.1, 0.17))
    logpsi = make_logpsi_slater(*occupied_orbital_matrix(g))
    Ed = make_local_energy(logpsi, 1.0, lr, hz, use_folx=False)
    Ef = make_local_energy(logpsi, 1.0, lr, hz, use_folx=True)
    rng = np.random.default_rng(0)
    for _ in range(8):
        R = jnp.asarray(rng.uniform(0, L, (N, 2))); S = jnp.asarray(rng.integers(0, 2, N))
        ed, ef = complex(Ed(R, S)), complex(Ef(R, S))
        assert abs(ef - ed) < 1e-9, f"folx != dense: {ef} vs {ed}"
        assert abs(ef.real - g["E_gs"]) < 1e-7, f"folx E_loc != E_gs ({ef.real} vs {g['E_gs']})"


def test_folx_matches_dense_psiformer():
    N, L, hz = 5, 7.0, 0.5
    g = noninteracting_gs(N=N, L=L, lambda_r=1.0, h_z=hz, n_cut=10, twist=(0.1, 0.17))
    Gs, W = planewave_envelope(*occupied_orbital_matrix(g))
    model = ContinuumSpinorPsiformer(n_elec=N, Gs=Gs, W=W, L=L, embed_dim=16, depth=2, n_heads=4)
    R0 = jnp.asarray(np.random.default_rng(0).uniform(0, L, (N, 2)))
    S0 = jnp.asarray(np.random.default_rng(1).integers(0, 2, N))
    # perturb backflow so the wavefunction is nontrivial (not just the exact Slater)
    params = model.init(jax.random.PRNGKey(0), R0, S0)
    key = jax.random.PRNGKey(5)
    params = jax.tree_util.tree_map(
        lambda x: x + 0.05 * (jax.random.normal(key, x.shape)
                              + (1j * jax.random.normal(key, x.shape) if jnp.iscomplexobj(x) else 0)).astype(x.dtype),
        params)
    logpsi = make_logpsi(model, params)
    Ed = make_local_energy(logpsi, 1.0, 1.0, hz, use_folx=False)
    Ef = make_local_energy(logpsi, 1.0, 1.0, hz, use_folx=True)
    rng = np.random.default_rng(3)
    for _ in range(6):
        R = jnp.asarray(rng.uniform(0, L, (N, 2))); S = jnp.asarray(rng.integers(0, 2, N))
        assert abs(complex(Ef(R, S)) - complex(Ed(R, S))) < 1e-9, "folx != dense (psiformer)"


def test_folx_matches_dense_spiral_with_extfield():
    res = spiral_gs(N=7, L=8.0, Q_int=(1, 0), B=0.3, lambda_r=1.0, h_z=0.0, n_cut=10, twist=(0.1, 0.17))
    logpsi = make_logpsi_general(res["Gs"], res["W"])
    ext = spiral_field(res["B"], res["Q"])
    Ed = make_local_energy(logpsi, 1.0, res["lambda_r"], 0.0, ext_field=ext, use_folx=False)
    Ef = make_local_energy(logpsi, 1.0, res["lambda_r"], 0.0, ext_field=ext, use_folx=True)
    rng = np.random.default_rng(2)
    for _ in range(6):
        R = jnp.asarray(rng.uniform(0, res["L"], (7, 2))); S = jnp.asarray(rng.integers(0, 2, 7))
        ed, ef = complex(Ed(R, S)), complex(Ef(R, S))
        assert abs(ef - ed) < 1e-9, f"folx != dense (spiral): {ef} vs {ed}"
        assert abs(ef.real - res["E_gs"]) < 1e-7, "folx spiral E_loc != E_gs"


def test_folx_works_under_vmap():
    N, L, hz = 5, 7.0, 0.5
    g = noninteracting_gs(N=N, L=L, lambda_r=1.0, h_z=hz, n_cut=10, twist=(0.1, 0.17))
    logpsi = make_logpsi_slater(*occupied_orbital_matrix(g))
    Ef = make_local_energy(logpsi, 1.0, 1.0, hz, use_folx=True)
    rng = np.random.default_rng(7)
    R = jnp.asarray(rng.uniform(0, L, (16, N, 2))); S = jnp.asarray(rng.integers(0, 2, (16, N)))
    ev = jax.vmap(Ef)(R, S)
    assert ev.shape == (16,)
    assert np.allclose(np.real(np.array(ev)), g["E_gs"], atol=1e-7), "vmapped folx != E_gs"


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
