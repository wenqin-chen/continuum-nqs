"""2D Ewald Coulomb on a GENERAL lattice cell (JAX) -- generalizes the square-box version.

Periodic Coulomb V(r)=alpha/r (2D FT = 2*pi*alpha/q), jellium background V(q=0)=0 -- the HF
convention (V(q)=2*pi*alpha/q). Ewald-split into a short-range
real-space erfc sum and a long-range reciprocal sum; works on ANY 2D Bravais cell (square box,
rectangular aspect-sqrt3, hexagonal/triangular) via the cell matrix A (columns = lattice vectors).

The reciprocal kernel is (2*pi/area) erfc(|G|/2eta)/|G| over the full +/-G set; with eta the
(arbitrary) Ewald width, the neutral-system energy is independent of eta (the correctness test).
(The square-box version had a factor-2 bug inherited from build_n09: pi/A vs the correct 2*pi/area.)

make_ewald_lattice(A, alpha, ...) -> (ewald_pair(r), zeta_M, coulomb_energy(R)).
make_ewald(L, alpha, ...) is the square special case A = L*I (UNCHANGED interface).
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import erfc as jerfc
from scipy.special import erfc as erfc_np

from .cell import square_cell, reciprocal, cell_area

jax.config.update("jax_enable_x64", True)


def make_ewald_lattice(A, alpha=1.0, ewald_alpha=None, n_real=5, n_recip=10):
    A = np.asarray(A, dtype=float)
    area = cell_area(A)
    B = reciprocal(A)                       # columns b1, b2 ; a_i . b_j = 2 pi delta
    eta = float(np.sqrt(np.pi) / np.sqrt(area)) if ewald_alpha is None else float(ewald_alpha)

    # real-space lattice images R = A @ (nx, ny)
    nx, ny = np.meshgrid(np.arange(-n_real, n_real + 1), np.arange(-n_real, n_real + 1), indexing="ij")
    cells = np.stack([nx.ravel(), ny.ravel()], axis=-1).astype(float)   # (n, 2)
    R_lat_np = cells @ A.T                                              # (n, 2)
    R_lat = jnp.asarray(R_lat_np)

    # reciprocal lattice G = B @ (mx, my), excluding G = 0
    mx, my = np.meshgrid(np.arange(-n_recip, n_recip + 1), np.arange(-n_recip, n_recip + 1), indexing="ij")
    mcell = np.stack([mx.ravel(), my.ravel()], axis=-1).astype(float)
    keep = (mcell[:, 0] ** 2 + mcell[:, 1] ** 2) > 0
    G_lat = jnp.asarray(mcell[keep] @ B.T)                              # (nG, 2)
    G_mag = jnp.linalg.norm(G_lat, axis=-1)
    RECIP = (2.0 * jnp.pi / area) * jerfc(G_mag / (2.0 * eta)) / G_mag
    V_G0 = -2.0 * jnp.sqrt(jnp.pi) / (area * eta)

    def ewald_pair(r):
        d = jnp.linalg.norm(r[None, :] + R_lat, axis=-1)
        V_real = jnp.sum(jerfc(eta * d) / d)
        V_recip = jnp.sum(RECIP * jnp.cos(G_lat @ r))
        return alpha * (V_real + V_recip + V_G0)

    # Madelung: each electron with its own images + the neutralizing background
    Rs = np.linalg.norm(R_lat_np, axis=-1)
    real_nz = float(sum(erfc_np(eta * d) / d for d in Rs if d > 1e-10))
    V_self = -2.0 * eta / np.sqrt(np.pi)
    V_recip0 = float(jnp.sum(RECIP))
    zeta_M = float(alpha * (real_nz + V_self + V_recip0 + float(V_G0)))

    def coulomb_energy(R):
        N = R.shape[0]
        dr = R[:, None, :] - R[None, :, :]
        iu = jnp.triu_indices(N, 1)
        pair = jax.vmap(ewald_pair)(dr[iu])
        return jnp.sum(pair) + N * zeta_M / 2.0

    return ewald_pair, zeta_M, coulomb_energy


def make_ewald(L, alpha=1.0, ewald_alpha=None, n_real=4, n_recip=8):
    """Square-box special case A = L*I (the original interface; tests use this)."""
    return make_ewald_lattice(square_cell(L), alpha=alpha, ewald_alpha=ewald_alpha,
                              n_real=n_real, n_recip=n_recip)
