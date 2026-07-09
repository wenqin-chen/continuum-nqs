"""Validate the production pieces: the base envelope on a non-square cell + the SkX pinning target.

  - cell_rashba_envelope on a RECTANGULAR cell gives zero-variance E_loc = E_gs (the full stack --
    general cell + envelope + SOC local energy -- is exact on the production geometry).
  - the analytic triple-Q SkX pinning target is TOPOLOGICAL (Berg-Luscher != 0) over the magnetic cell.

Run:  python tests/test_nqs_production.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from src.nqs.cell import rect_cell
from src.nqs.rashba_reference import cell_rashba_envelope
from src.nqs.slater_reference import make_logpsi_general
from src.nqs.local_energy import make_local_energy
from src.nqs.pinning import make_skx_pin, make_fm_pin, make_spiral_pin, berg_luscher_realspace


def test_cell_rashba_envelope_zero_variance_rect():
    A = rect_cell(8.0, 8.0 * np.sqrt(3))
    N, lr, hz = 6, 1.0, 0.5
    env = cell_rashba_envelope(A, N, lambda_r=lr, h_z=hz, n_cut=8, twist=(0.1, 0.17))
    logpsi = make_logpsi_general(env["Gs"], env["W"])
    E_loc = make_local_energy(logpsi, 1.0, lr, hz)
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(12):
        R = jnp.asarray(rng.uniform(0, 1, (N, 2)) @ np.asarray(A).T)
        S = jnp.asarray(rng.integers(0, 2, N))
        vals.append(complex(E_loc(R, S)))
    vals = np.array(vals)
    assert np.std(vals.real) < 1e-7, f"base envelope not zero-variance on rect cell: {np.std(vals.real):.2e}"
    assert abs(vals.real.mean() - env["E_gs"]) < 1e-7, \
        f"<E_loc>={vals.real.mean():.8f} != E_gs={env['E_gs']:.8f}"


def _mag_cell(qstar):
    q1 = qstar * np.array([1.0, 0.0])
    q2 = qstar * np.array([-0.5, np.sqrt(3) / 2])
    A_mag = 2 * np.pi * np.linalg.inv(np.array([q1, q2]))     # columns dual to q1,q2
    return q1, q2, A_mag


def test_skx_pin_is_topological():
    q1, q2, A_mag = _mag_cell(0.9548)
    pin = make_skx_pin(q1, q2, h_pin=1.0, m0=0.3)
    bl = berg_luscher_realspace(pin, A_mag, ngrid=36)
    print(f"  skx target BL = {bl:+.3f}")
    assert abs(bl) > 0.5, f"skx target not topological (BL={bl:.3f}); tune m0"


def test_pins_return_unit_fields():
    q1, q2, _ = _mag_cell(0.9548)
    r = jnp.asarray([1.3, -0.7])
    for pin in (make_skx_pin(q1, q2, 0.7), make_fm_pin(0.7), make_spiral_pin(q1, 0.7)):
        f = np.array(pin(r))
        assert f.shape == (3,) and abs(np.linalg.norm(f) - 0.7) < 1e-6, "pin field not h_pin*unit"


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
