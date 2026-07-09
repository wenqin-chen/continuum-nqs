"""Exact non-interacting SPIN-SPIRAL ground state -- a non-collinear validation oracle.

Single-particle Hamiltonian = the Rashba 2DEG + a helical (spiral) in-plane Zeeman field of
wavevector Q:

    H = sum_i [ h0(k_i)  -  B ( sigma_x cos(Q.r_i) + sigma_y sin(Q.r_i) ) ]

with h0 the usual kinetic + Rashba + Zeeman. The spiral field
    -B(sigma_x cos + sigma_y sin) = -B( sigma_- e^{iQ.r} + sigma_+ e^{-iQ.r} )
couples plane waves |k,up> <-> |k+Q,down>, so each single-particle eigenstate is a COMB
{k0 + nQ} with a winding (non-collinear) spinor. The N-electron ground state is a single Slater
determinant of the lowest N such combs -- exactly solvable, and a genuinely non-collinear test of
the SOC + multi-G-envelope machinery (mirrors the spin-spiral validation in arXiv:2510.18621).

`spiral_gs` diagonalises H in a finite plane-wave basis and returns the exact E_gs together with
the occupied orbitals as a multi-G envelope (Gs, W) ready for make_logpsi_general / the Psiformer.
`spiral_field` is the matching ext_field for the local energy.
"""
import numpy as np
import jax.numpy as jnp
from .rashba_reference import h0_k


def spiral_gs(N, L, Q_int, B, m_star=1.0, lambda_r=1.0, h_z=0.0, n_cut=6, twist=(0.0, 0.0)):
    """Exact spiral ground state. Q_int=(qx,qy) integers -> Q=(2pi/L)(qx,qy) commensurate.
    Returns dict(E_gs, gap, Gs (n_pw,2), W (N,n_pw,2), Q, boundary_weight, cutoff_ok, ...)."""
    qx, qy = Q_int
    Qx, Qy = 2.0 * np.pi * qx / L, 2.0 * np.pi * qy / L
    js = np.arange(-n_cut, n_cut + 1)
    grid = [(nx, ny) for nx in js for ny in js]
    idx = {nm: p for p, nm in enumerate(grid)}
    npw = len(grid)
    kx = np.array([(2 * np.pi * nx + twist[0]) / L for (nx, ny) in grid])
    ky = np.array([(2 * np.pi * ny + twist[1]) / L for (nx, ny) in grid])

    H = np.zeros((2 * npw, 2 * npw), dtype=complex)
    for p in range(npw):
        H[2 * p:2 * p + 2, 2 * p:2 * p + 2] = h0_k(kx[p], ky[p], m_star, lambda_r, h_z)
    # spiral coupling: (k, up=0) <-> (k+Q, down=1) with amplitude -B (+ Hermitian conjugate)
    for p, (nx, ny) in enumerate(grid):
        key = (nx + qx, ny + qy)
        if key in idx:
            p2 = idx[key]
            H[2 * p2 + 1, 2 * p + 0] += -B
            H[2 * p + 0, 2 * p2 + 1] += -B

    E, V = np.linalg.eigh(H)                          # ascending
    if 2 * npw < N + 1:
        raise ValueError("basis too small for N")
    occ = V[:, :N]                                    # (2 npw, N)
    E_gs = float(np.sum(E[:N]))
    gap = float(E[N] - E[N - 1])
    Gs = np.stack([kx, ky], axis=-1)                  # (npw, 2)
    W = np.zeros((N, npw, 2), dtype=complex)
    for a in range(N):
        W[a] = occ[:, a].reshape(npw, 2)              # W[a,g,sigma] = <g,sigma|orbital_a>
    # basis-completeness: occupied weight that leaks to the grid boundary
    bmask = np.array([(abs(nx) == n_cut or abs(ny) == n_cut) for (nx, ny) in grid])
    bweight = float(np.sum(np.abs(occ.reshape(npw, 2, N)[bmask]) ** 2))
    # the spiral field couples each comb's boundary plane wave to one OUTSIDE the basis, so the
    # truncated eigenstate is an exact continuum eigenstate only when this leakage is negligible.
    # E_loc std ~ B*sqrt(boundary_weight); require < 1e-13 so it predicts machine-precision zero var.
    return dict(E_gs=E_gs, gap=gap, Gs=Gs, W=W, N=N, L=L, Q=(Qx, Qy), Q_int=Q_int, B=B,
                m_star=m_star, lambda_r=lambda_r, h_z=h_z, twist=twist,
                boundary_weight=bweight, cutoff_ok=bool(bweight < 1e-13))


def spiral_field(B, Q):
    """ext_field(r) = (B cos(Q.r), B sin(Q.r), 0) for make_local_energy (JAX)."""
    Qx, Qy = Q

    def f(r):
        ph = Qx * r[0] + Qy * r[1]
        return jnp.array([B * jnp.cos(ph), B * jnp.sin(ph), 0.0])
    return f


def spin_texture(res, ngrid=12):
    """Real-space <sigma(r)> of the occupied determinant on an ngrid x ngrid mesh of the box.
    Returns S (ngrid,ngrid,3) real -- for checking the texture is genuinely non-collinear."""
    Gs, W, L = res["Gs"], res["W"], res["L"]
    xs = (np.arange(ngrid) + 0.5) * L / ngrid
    Sx = np.array([[1, 0], [0, 1]])  # placeholder not used
    SX = np.array([[0, 1], [1, 0]], complex)
    SY = np.array([[0, -1j], [1j, 0]], complex)
    SZ = np.array([[1, 0], [0, -1]], complex)
    S = np.zeros((ngrid, ngrid, 3))
    for ix, x in enumerate(xs):
        for iy, y in enumerate(xs):
            ph = np.exp(1j * (Gs[:, 0] * x + Gs[:, 1] * y))        # (npw,)
            # orbital spinors at r: phi[a] = sum_g W[a,g,:] ph[g]  -> (N,2)
            phi = np.einsum("agc,g->ac", W[:], ph)
            for M, k in ((SX, 0), (SY, 1), (SZ, 2)):
                S[ix, iy, k] = np.real(np.sum(np.einsum("ac,cd,ad->a", phi.conj(), M, phi)))
    return S
