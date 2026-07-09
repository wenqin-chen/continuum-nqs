"""NON-COLLINEAR validation gate: the exact spin-spiral ground state.

  (1) ZERO VARIANCE: for the exact occupied determinant of (Rashba + spiral field), the full
      local energy (kinetic + SOC + spiral ext_field) equals E_gs at every (R,S) -- certifies
      the ext_field term and the multi-G envelope on a genuinely non-collinear state.
  (2) NON-COLLINEAR: the real-space <sigma(r)> winds (S_x and S_y both change sign).
  (3) PSIFORMER: zero-init backflow on the spiral (Gs,W) reproduces E_gs to machine precision
      (the multi-G warm-start the SkX will use).

Run:  python tests/test_nqs_spiral.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from src.nqs.spiral_reference import spiral_gs, spiral_field, spin_texture
from src.nqs.slater_reference import make_logpsi_general
from src.nqs.local_energy import make_local_energy
from src.nqs.psiformer import ContinuumSpinorPsiformer, make_logpsi

PARAMS = dict(N=7, L=8.0, Q_int=(1, 0), B=0.3, lambda_r=1.0, h_z=0.0,
              n_cut=10, twist=(0.1, 0.17))


def _setup():
    res = spiral_gs(**PARAMS)
    logpsi = make_logpsi_general(res["Gs"], res["W"])
    ext = spiral_field(res["B"], res["Q"])
    E_loc = make_local_energy(logpsi, m_star=1.0, lambda_r=res["lambda_r"],
                              h_z=res["h_z"], ext_field=ext)
    return res, logpsi, E_loc


def test_basis_converged_and_gapped():
    res = spiral_gs(**PARAMS)
    assert res["cutoff_ok"], f"occupied weight leaks to boundary ({res['boundary_weight']:.1e})"
    assert res["gap"] > 1e-6, f"spiral GS not gapped (gap={res['gap']:.2e}); pick another N/twist"


def test_spiral_zero_variance_equals_Egs():
    res, logpsi, E_loc = _setup()
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(16):
        R = jnp.asarray(rng.uniform(0, res["L"], (res["N"], 2)))
        S = jnp.asarray(rng.integers(0, 2, res["N"]))
        vals.append(complex(E_loc(R, S)))
    vals = np.array(vals)
    assert np.std(vals.real) < 1e-7, f"E_loc not zero-variance on spiral: std {np.std(vals.real):.2e}"
    assert abs(vals.real.mean() - res["E_gs"]) < 1e-7, \
        f"<E_loc>={vals.real.mean():.8f} != E_gs={res['E_gs']:.8f}"
    assert np.max(np.abs(vals.imag)) < 1e-7, "Im(E_loc) not ~0"


def test_texture_is_noncollinear():
    res = spiral_gs(**PARAMS)
    S = spin_texture(res, ngrid=12)
    Sx, Sy = S[..., 0], S[..., 1]
    assert Sx.max() > 1e-3 and Sx.min() < -1e-3, "S_x does not wind (collinear?)"
    assert Sy.max() > 1e-3 and Sy.min() < -1e-3, "S_y does not wind (collinear?)"


def test_psiformer_zero_backflow_on_spiral():
    res = spiral_gs(**PARAMS)
    N, L = res["N"], res["L"]
    model = ContinuumSpinorPsiformer(n_elec=N, Gs=res["Gs"], W=res["W"], L=L,
                                     embed_dim=16, depth=2, n_heads=4)
    R0 = jnp.asarray(np.random.default_rng(0).uniform(0, L, (N, 2)))
    S0 = jnp.asarray(np.random.default_rng(1).integers(0, 2, N))
    params = model.init(jax.random.PRNGKey(0), R0, S0)
    logpsi = make_logpsi(model, params)
    ext = spiral_field(res["B"], res["Q"])
    E_loc = make_local_energy(logpsi, m_star=1.0, lambda_r=res["lambda_r"],
                              h_z=res["h_z"], ext_field=ext)
    rng = np.random.default_rng(3)
    vals = [complex(E_loc(jnp.asarray(rng.uniform(0, L, (N, 2))),
                          jnp.asarray(rng.integers(0, 2, N)))) for _ in range(8)]
    vals = np.array(vals)
    assert np.std(vals.real) < 1e-6, f"Psiformer not zero-variance on spiral: {np.std(vals.real):.2e}"
    assert abs(vals.real.mean() - res["E_gs"]) < 1e-6, \
        f"Psiformer init E={vals.real.mean():.8f} != E_gs={res['E_gs']:.8f}"


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
    res = spiral_gs(**PARAMS)
    print(f"  spiral GS: N={res['N']} L={res['L']} Q_int={res['Q_int']} B={res['B']} "
          f"lr={res['lambda_r']}: E_gs={res['E_gs']:+.6f} gap={res['gap']:.4f} "
          f"cutoff_ok={res['cutoff_ok']}")
