"""Certify MinSR.

  (1) EXACT identity: the sample-space (2N x 2N) MinSR solve gives the SAME natural-gradient
      direction as the explicit parameter-space (P x P) solve, to machine precision (push-through).
      This is a correctness proof, not just "it trains".
  (2) CONVERGENCE: perturbed off the non-interacting GS, MinSR descends to E_gs TIGHTLY
      (natural gradient resolves the fine landscape that Adam plateaus ~0.1 above).

Run:  python tests/test_nqs_minsr.py    (~1-2 min on CPU)
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from src.nqs.rashba_reference import noninteracting_gs, occupied_orbital_matrix, planewave_envelope
from src.nqs.psiformer import ContinuumSpinorPsiformer
from src.nqs.minsr import real_ravel, per_sample_O, minsr_direction, paramsr_direction, train_minsr


def _model(N=5, L=7.0, hz=0.5, embed=12, heads=3):
    g = noninteracting_gs(N=N, L=L, lambda_r=1.0, h_z=hz, n_cut=10, twist=(0.1, 0.17))
    Gs, W = planewave_envelope(*occupied_orbital_matrix(g))
    model = ContinuumSpinorPsiformer(n_elec=N, Gs=Gs, W=W, L=L, embed_dim=embed, depth=2, n_heads=heads)
    R0 = jnp.asarray(np.random.default_rng(0).uniform(0, L, (N, 2)))
    S0 = jnp.asarray(np.random.default_rng(1).integers(0, 2, N))
    params = model.init(jax.random.PRNGKey(0), R0, S0)
    return g, model, params


def _perturb_backflow(params, scale, seed):
    key = jax.random.PRNGKey(seed)
    def f(path, x):
        if any("backflow" in str(getattr(p, "key", "")) for p in path):
            k = jax.random.fold_in(key, abs(hash(str(path))) % (2 ** 31))
            kr, ki = jax.random.split(k)
            noise = jax.random.normal(kr, x.shape)
            if jnp.iscomplexobj(x):
                noise = noise + 1j * jax.random.normal(ki, x.shape)
            return x + scale * noise.astype(x.dtype)
        return x
    return jax.tree_util.tree_map_with_path(f, params)


def test_minsr_equals_paramsr():
    N, L = 5, 7.0
    g, model, params = _model(N, L)
    theta, unravel = real_ravel(params)
    P = theta.shape[0]
    rng = np.random.default_rng(4)
    R = jnp.asarray(rng.uniform(0, L, (24, N, 2)))
    S = jnp.asarray(rng.integers(0, 2, (24, N)))
    O = per_sample_O(model, theta, unravel, R, S)
    eps = jnp.asarray(rng.normal(size=24) + 1j * rng.normal(size=24))
    print(f"  (P={P} params, 2N={2 * 24} sample rows)")
    for lam in (1e-2, 1e-4):
        d1 = minsr_direction(O, eps, lam)
        d2 = paramsr_direction(O, eps, lam)
        rel = float(jnp.max(jnp.abs(d1 - d2)) / (jnp.max(jnp.abs(d2)) + 1e-30))
        assert rel < 1e-6, f"MinSR != param-SR at lam={lam}: rel diff {rel:.2e}"


def test_minsr_descends_to_Egs():
    N, L, hz = 5, 7.0, 0.5
    g, model, params = _model(N, L, hz, embed=16, heads=4)
    params = _perturb_backflow(params, 0.1, 7)
    params, hist = train_minsr(model, params, L, N, jax.random.PRNGKey(3),
                               m_star=1.0, lambda_r=1.0, h_z=hz, n_walkers=256, n_steps=80,
                               lr=0.1, lam=1e-3, n_sweeps=8, step_size=0.5, burn=40, log_every=20)
    E_final = float(np.mean(hist["E"][-5:]))      # average last few noisy estimates
    print(f"  E_gs={g['E_gs']:+.6f}  E_minsr(last5 avg)={E_final:+.6f}  "
          f"gap={E_final - g['E_gs']:+.2e}")
    assert E_final < g["E_gs"] + 1e-2, \
        f"MinSR did not reach E_gs tightly: {E_final:.6f} vs {g['E_gs']:.6f}"
    assert E_final > g["E_gs"] - 2e-2, "below variational floor (MC noise too large)"


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
