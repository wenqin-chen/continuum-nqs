"""MinSR (minimum-step Stochastic Reconfiguration) natural-gradient optimizer (JAX).

Stochastic Reconfiguration preconditions the energy gradient g by the inverse quantum metric S:
    dtheta = (S + lam I)^{-1} g,   S_kl = Re<dO_k* dO_l>,  g_k = 2 Re<dO_k* eps>,
with O_sk = d logPsi(x_s)/d theta_k (real params), dO centered, eps = E_loc - <E>.

S is P x P (P = #params, large for a net). MinSR (Chen-Heyl 2023) uses the EXACT identity
    (Otil^T Otil + lam I)^{-1} Otil^T  ==  Otil^T (Otil Otil^T + lam I)^{-1}
(push-through), so the natural gradient can be solved in the SAMPLE space (2N x 2N) instead of
parameter space (P x P) -- a huge win when 2N << P. Here Otil = [Re dO; Im dO] (2N x P real),
etil = [Re eps; Im eps] (2N), and dtheta = Otil^T (Otil Otil^T + lam I)^{-1} etil.

Because the identity is EXACT, MinSR is validated to machine precision against the explicit
param-space solve -- not merely "it trains". Complex params are handled by a real ravel
(split re/im), so all SR linear algebra is real and Wirtinger-free.
"""
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

jax.config.update("jax_enable_x64", True)

from .local_energy import make_local_energy
from .sampler import make_mcmc, _cell_matrix
from .psiformer import make_logpsi


# --- real ravel of a (possibly complex) param pytree ------------------------------------
def real_ravel(pytree):
    """Flatten a pytree to a REAL vector (complex leaves -> [re, im]); return (vec, unravel)."""
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    specs, flats = [], []
    for l in leaves:
        if jnp.iscomplexobj(l):
            specs.append((True, l.shape, l.size))
            flats.append(jnp.real(l).ravel()); flats.append(jnp.imag(l).ravel())
        else:
            specs.append((False, l.shape, l.size))
            flats.append(l.ravel())
    vec = jnp.concatenate(flats)

    def unravel(v):
        out, i = [], 0
        for is_c, shape, size in specs:
            if is_c:
                re = v[i:i + size].reshape(shape); i += size
                im = v[i:i + size].reshape(shape); i += size
                out.append(re + 1j * im)
            else:
                out.append(v[i:i + size].reshape(shape)); i += size
        return jax.tree_util.tree_unflatten(treedef, out)
    return vec, unravel


def per_sample_O(model, theta, unravel, R, S):
    """O[s,k] = d logPsi(x_s)/d theta_k (complex; real params). Returns (N, P) complex."""
    def lr(t, r, s):
        return jnp.real(model.apply(unravel(t), r, s))

    def li(t, r, s):
        return jnp.imag(model.apply(unravel(t), r, s))
    Jre = jax.vmap(lambda r, s: jax.grad(lr)(theta, r, s))(R, S)
    Jim = jax.vmap(lambda r, s: jax.grad(li)(theta, r, s))(R, S)
    return Jre + 1j * Jim


def _otil_etil(O, eps):
    dO = O - jnp.mean(O, axis=0, keepdims=True)
    Otil = jnp.concatenate([jnp.real(dO), jnp.imag(dO)], axis=0)        # (2N, P)
    ec = eps - jnp.mean(eps)
    etil = jnp.concatenate([jnp.real(ec), jnp.imag(ec)])               # (2N,)
    return Otil, etil


def minsr_direction(O, eps, lam):
    """Natural-gradient direction via the SAMPLE-space (2N x 2N) solve."""
    Otil, etil = _otil_etil(O, eps)
    T = Otil @ Otil.T
    y = jnp.linalg.solve(T + lam * jnp.eye(T.shape[0]), etil)
    return Otil.T @ y


def paramsr_direction(O, eps, lam):
    """Same natural gradient via the explicit PARAMETER-space (P x P) solve (reference)."""
    Otil, etil = _otil_etil(O, eps)
    P = Otil.shape[1]
    return jnp.linalg.solve(Otil.T @ Otil + lam * jnp.eye(P), Otil.T @ etil)


# --- MinSR training loop ----------------------------------------------------------------
def train_minsr(model, params, L, n_elec, key, *, m_star=1.0, lambda_r=1.0, h_z=0.0,
                ext_field=None, coulomb=None, use_folx=False, n_walkers=256, n_steps=100,
                lr=0.05, lam=1e-3, n_sweeps=10, step_size=0.4, burn=80, clip_k=5.0,
                log_every=20, history=None, R0=None, S0=None, return_walkers=False):
    """Optimize the Psiformer with MinSR. Returns (params, history), or
    (params, history, R, S) if return_walkers=True.

    R0/S0 (optional): warm-start the walker ensemble (e.g. from a previous training segment).
    Cold init (R0=None) draws uniform walkers + `burn` thermalization sweeps; a warm ensemble is
    already ~|Psi|^2-distributed, so the burn stage is SKIPPED (the per-step n_sweeps decorrelate).
    This removes the fresh-chain segment sawtooth (each plateau segment
    used to re-thermalize from uniform, biasing early-segment energies high for crystal states)."""
    theta, unravel = real_ravel(params)
    sweep = make_mcmc(make_logpsi(model, params), L, n_elec)

    def sample(theta, R, S, key, n):
        lp = make_logpsi(model, unravel(theta))
        sw = make_mcmc(lp, L, n_elec)
        vsw = jax.vmap(sw, in_axes=(0, 0, 0, None))

        def body(carry, k):
            R, S = carry
            R, S, ap, _ = vsw(jax.random.split(k, R.shape[0]), R, S, step_size)
            return (R, S), ap
        (R, S), aps = jax.lax.scan(body, (R, S), jax.random.split(key, n))
        return R, S, jnp.mean(aps)
    sample = jax.jit(sample, static_argnums=(4,))

    @jax.jit
    def energy_and_O(theta, R, S):
        lp = make_logpsi(model, unravel(theta))
        E_loc = make_local_energy(lp, m_star, lambda_r, h_z, ext_field=ext_field, use_folx=use_folx)
        e = jax.vmap(E_loc)(R, S)
        if coulomb is not None:
            e = e + jax.vmap(coulomb)(R)
        O = per_sample_O(model, theta, unravel, R, S)
        return e, O

    kk, ks = jax.random.split(key)
    A_cell, _ = _cell_matrix(L)                                  # L = scalar (square) or 2x2 cell
    if R0 is None:
        R = jax.random.uniform(kk, (n_walkers, n_elec, 2)) @ A_cell.T
        S = jax.random.randint(jax.random.fold_in(kk, 1), (n_walkers, n_elec), 0, 2)
        R, S, _ = sample(theta, R, S, ks, burn)
    else:
        R, S = jnp.asarray(R0), jnp.asarray(S0)                  # warm ensemble: no burn stage
    hist = history if history is not None else {"E": [], "Eerr": []}
    for it in range(n_steps):
        R, S, ap = sample(theta, R, S, jax.random.fold_in(ks, 1000 + it), n_sweeps)
        e, O = energy_and_O(theta, R, S)
        # median-MAD clip of local-energy OUTLIERS (e.g. Coulomb cusp spikes when two electrons
        # approach without a cusp Jastrow) before forming the gradient -- Lesson #3. Essential for
        # the interacting problem; a no-op for zero-variance non-interacting states.
        e_re = jnp.real(e)
        med = jnp.median(e_re)
        mad = jnp.median(jnp.abs(e_re - med)) + 1e-12
        e_clip = jnp.clip(e_re, med - clip_k * mad, med + clip_k * mad) + 1j * jnp.imag(e)
        Ebar = jnp.real(jnp.mean(e))            # UNCLIPPED variational energy (honest report)
        d = minsr_direction(O, e_clip, lam)     # clip ONLY the gradient (stability), not the energy
        theta = theta - lr * d
        Eerr = jnp.real(jnp.std(e_re) / jnp.sqrt(e.shape[0]))
        hist["E"].append(float(Ebar)); hist["Eerr"].append(float(Eerr))
        if log_every and (it % log_every == 0 or it == n_steps - 1):
            print(f"    minsr {it:4d}: E={float(Ebar):+.6f} +/- {float(Eerr):.1e} "
                  f"acc={float(ap):.2f}", flush=True)
    if return_walkers:
        return unravel(theta), hist, R, S
    return unravel(theta), hist
