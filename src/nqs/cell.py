"""General 2D simulation cell for the NQS -- generalizes the square box L to any lattice.

A cell is a 2x2 matrix A whose COLUMNS are the lattice vectors a1, a2: a cartesian point is
r = A @ s with fractional coordinates s in [0,1)^2. Reciprocal B = 2*pi (A^-1)^T (columns b1,b2),
satisfying a_i . b_j = 2*pi delta_ij. The square box of side L is A = L * I.

Covers rectangular supercells (aspect sqrt3, tiles the triangular magnetic lattice) for
the G2 energy run, and the full hexagonal/triangular cell for the G3 <C6> topology.
"""
import numpy as np
import jax.numpy as jnp


def square_cell(L):
    return np.array([[float(L), 0.0], [0.0, float(L)]])


def rect_cell(Lx, Ly):
    return np.array([[float(Lx), 0.0], [0.0, float(Ly)]])


def hex_cell(a):
    """Triangular lattice, constant a: a1=(a,0), a2=(a/2, a*sqrt3/2) (60-deg cell)."""
    a = float(a)
    return np.array([[a, a / 2.0], [0.0, a * np.sqrt(3) / 2.0]])


def reciprocal(A):
    """B = 2*pi (A^-1)^T; columns are b1,b2 with a_i . b_j = 2*pi delta_ij."""
    return 2.0 * np.pi * np.linalg.inv(np.asarray(A, float)).T


def cell_area(A):
    return float(abs(np.linalg.det(np.asarray(A, float))))


def frac(r, A):
    """Cartesian -> fractional coords mod 1 (JAX). r (...,2)."""
    Ainv = jnp.linalg.inv(jnp.asarray(A))
    return jnp.mod(r @ Ainv.T, 1.0)


def wrap(r, A):
    """Wrap a cartesian point into the cell (JAX). r (...,2)."""
    Aj = jnp.asarray(A)
    return frac(r, Aj) @ Aj.T
