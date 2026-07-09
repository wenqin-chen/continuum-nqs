"""VMC training loop for the continuum spinor Psiformer (JAX/optax).

One jitted step (params TRACED throughout, so no recompilation across steps):
  1. advance the persistent walker ensemble by `n_sweeps` Metropolis sweeps on |Psi_params|^2;
  2. evaluate the (validated) local energy E_loc on the walkers;
  3. form the VMC energy gradient via the SURROGATE loss
        surrogate(theta) = 2 * mean_w Re[ stopgrad(conj(E_loc - <E>)) * logPsi_theta(R_w,S_w) ]
     whose autodiff gradient equals the exact estimator  g_theta = 2 Re < (E_loc-<E>)^* d_theta logPsi >
     (correct for real backbone params AND complex backflow params under jax's Wirtinger
     convention -- the GS-recovery test certifies the sign);
  4. optax update.

The optimizer is pluggable (Adam by default; swap in MinSR/SPRING/KFAC for production -- Adam
will not reach sub-mHa, but it is sufficient to certify the loop against a known minimum).
"""
import jax
import jax.numpy as jnp
import optax
from functools import partial

jax.config.update("jax_enable_x64", True)

from .local_energy import make_local_energy
from .sampler import make_mcmc
from .psiformer import make_logpsi


def make_train_step(model, L, n_elec, m_star=1.0, lambda_r=1.0, h_z=0.0,
                    coulomb=None, optimizer=None, n_sweeps=10, step_size=0.4,
                    n_spin_flips=2):
    """Return (optimizer, jitted step(params, opt_state, R, S, key))."""
    if optimizer is None:
        optimizer = optax.adam(1e-3)

    @jax.jit
    def step(params, opt_state, R, S, key):
        logpsi = make_logpsi(model, params)              # closes over TRACED params
        sweep = make_mcmc(logpsi, L, n_elec, n_spin_flips)
        vsweep = jax.vmap(sweep, in_axes=(0, 0, 0, None))

        def mc_body(carry, k):
            R, S = carry
            keys = jax.random.split(k, R.shape[0])
            R, S, ap, asp = vsweep(keys, R, S, step_size)
            return (R, S), (jnp.mean(ap), jnp.mean(asp))
        (R, S), (ap, asp) = jax.lax.scan(mc_body, (R, S), jax.random.split(key, n_sweeps))

        E_loc = make_local_energy(logpsi, m_star, lambda_r, h_z)
        e = jax.vmap(E_loc)(R, S)                         # (W,) complex
        if coulomb is not None:
            e = e + jax.vmap(coulomb)(R)                  # spin-independent pair energy
        Ebar = jnp.mean(e)
        ec = jax.lax.stop_gradient(jnp.conj(e - Ebar))

        def surrogate(p):
            lp = jax.vmap(make_logpsi(model, p))(R, S)    # (W,) complex
            return 2.0 * jnp.mean(jnp.real(ec * lp))

        g = jax.grad(surrogate)(params)
        updates, opt_state = optimizer.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        Eerr = jnp.std(jnp.real(e)) / jnp.sqrt(e.shape[0])
        return params, opt_state, R, S, jnp.real(Ebar), Eerr, jnp.mean(ap), jnp.mean(asp)

    return optimizer, step


def init_walkers(key, n_walkers, n_elec, L):
    k0, k1 = jax.random.split(key)
    R = jax.random.uniform(k0, (n_walkers, n_elec, 2)) * L
    S = jax.random.randint(k1, (n_walkers, n_elec), 0, 2)
    return R, S


def train(model, params, L, n_elec, key, *, m_star=1.0, lambda_r=1.0, h_z=0.0,
          coulomb=None, n_walkers=256, n_steps=200, lr=1e-3, n_sweeps=10,
          step_size=0.4, burn=50, optimizer=None, log_every=20, history=None):
    """Run VMC training; return (params, history dict). history['E'] is the energy trace."""
    optimizer, step = make_train_step(model, L, n_elec, m_star, lambda_r, h_z, coulomb,
                                      optimizer or optax.adam(lr), n_sweeps, step_size)
    opt_state = optimizer.init(params)
    kw, ks = jax.random.split(key)
    R, S = init_walkers(kw, n_walkers, n_elec, L)
    # burn-in at the initial params
    for i in range(burn // n_sweeps + 1):
        params0, opt0, R, S, *_ = step(params, opt_state, R, S, jax.random.fold_in(ks, i))
    hist = history if history is not None else {"E": [], "Eerr": [], "acc_pos": []}
    for it in range(n_steps):
        params, opt_state, R, S, E, Eerr, ap, asp = step(
            params, opt_state, R, S, jax.random.fold_in(ks, 1000 + it))
        hist["E"].append(float(E)); hist["Eerr"].append(float(Eerr))
        hist["acc_pos"].append(float(ap))
        if log_every and (it % log_every == 0 or it == n_steps - 1):
            print(f"    step {it:4d}: E={float(E):+.6f} +/- {float(Eerr):.1e}  "
                  f"acc_pos={float(ap):.2f} acc_spin={float(asp):.2f}", flush=True)
    return params, hist
