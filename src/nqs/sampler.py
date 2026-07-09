"""Metropolis MCMC over (positions, spins) on the torus, sampling |Psi(R,S)|^2 (JAX, batched).

Generic in `logpsi(R,S)->complex`: drives the exact Slater trial today and the Psiformer later.
A sweep = one joint all-electron Gaussian position proposal (PBC-wrapped) + `n_spin` single-spin
flips, each Metropolis-accepted on |Psi|^2 = exp(2 Re logPsi). Positions live on [0,L)^2; wrapping
electron i by L multiplies its determinant column by a uniform twist phase, leaving |Psi|^2
invariant, so PBC wrapping is exact.
"""
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

jax.config.update("jax_enable_x64", True)

from .cell import square_cell


def _cell_matrix(cell):
    """Accept a scalar box side L (square) or a 2x2 lattice matrix A; return (A, A^-1) as jnp."""
    A = square_cell(cell) if np.ndim(cell) == 0 else np.asarray(cell, float)
    return jnp.asarray(A), jnp.asarray(np.linalg.inv(A))


def make_mcmc(logpsi, cell, n_elec, n_spin_flips=2):
    """Return sweep(key, R, S, step) -> (R', S', acc_pos, acc_spin) for ONE walker.
    `cell` = scalar box side L (square) or a 2x2 lattice matrix A (rect/hex supercell)."""
    A, Ainv = _cell_matrix(cell)

    def _wrap(r):                                     # wrap cartesian into the cell (PBC)
        return jnp.mod(r @ Ainv.T, 1.0) @ A.T

    def logp(R, S):
        return 2.0 * jnp.real(logpsi(R, S))

    def sweep(key, R, S, step):
        kpos, kpacc, kspin = jax.random.split(key, 3)
        # --- joint position move ---
        Rp = _wrap(R + step * jax.random.normal(kpos, (n_elec, 2)))
        d = logp(Rp, S) - logp(R, S)
        acc = jax.random.uniform(kpacc) < jnp.exp(jnp.minimum(0.0, d))
        R = jnp.where(acc, Rp, R)
        acc_pos = acc.astype(jnp.float64)

        # --- n_spin single-electron spin flips ---
        def flip(carry, k):
            S, nacc = carry
            ki, kf, ka = jax.random.split(k, 3)
            i = jax.random.randint(ki, (), 0, n_elec)
            Sp = S.at[i].set(1 - S[i])
            d = logp(R, Sp) - logp(R, S)
            a = jax.random.uniform(ka) < jnp.exp(jnp.minimum(0.0, d))
            S = jax.lax.cond(a, lambda: Sp, lambda: S)
            return (S, nacc + a.astype(jnp.float64)), None

        (S, nacc), _ = jax.lax.scan(flip, (S, 0.0), jax.random.split(kspin, n_spin_flips))
        return R, S, acc_pos, nacc / n_spin_flips

    return sweep


def run_chain(logpsi, cell, n_elec, key, n_walkers=256, n_sweeps=200, burn=100,
              step=0.4, n_spin_flips=2, R0=None, S0=None):
    """Run batched MCMC; return (R, S, accept_pos, accept_spin) AFTER burn-in.
    `cell` = scalar box side L (square) or 2x2 lattice matrix A. Samples the final-sweep ensemble."""
    A, _ = _cell_matrix(cell)
    sweep = make_mcmc(logpsi, cell, n_elec, n_spin_flips)
    vsweep = jax.vmap(sweep, in_axes=(0, 0, 0, None))
    k0, k1, k2 = jax.random.split(key, 3)
    if R0 is None:
        R = jax.random.uniform(k0, (n_walkers, n_elec, 2)) @ A.T
    else:
        R = jnp.broadcast_to(R0, (n_walkers, n_elec, 2)).copy()
    if S0 is None:
        S = jax.random.randint(k1, (n_walkers, n_elec), 0, 2)
    else:
        S = jnp.broadcast_to(S0, (n_walkers, n_elec)).copy()

    @partial(jax.jit, static_argnums=(3,))
    def scan_sweeps(R, S, key, n):
        def body(carry, k):
            R, S = carry
            keys = jax.random.split(k, R.shape[0])
            R, S, ap, asp = vsweep(keys, R, S, step)
            return (R, S), (jnp.mean(ap), jnp.mean(asp))
        (R, S), (ap, asp) = jax.lax.scan(body, (R, S), jax.random.split(key, n))
        return R, S, ap, asp

    R, S, _, _ = scan_sweeps(R, S, k2, burn)
    R, S, ap, asp = scan_sweeps(R, S, jax.random.fold_in(k2, 1), n_sweeps)
    return R, S, float(jnp.mean(ap)), float(jnp.mean(asp))


def vmc_energy(E_loc, R, S):
    """Mean local energy + standard error over a walker ensemble (R (W,N,2), S (W,N))."""
    e = jax.vmap(E_loc)(R, S)
    e = jnp.real(e)
    return float(jnp.mean(e)), float(jnp.std(e) / jnp.sqrt(e.shape[0])), e
