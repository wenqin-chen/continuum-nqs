"""Topology diagnostics for Stage G (single-particle ORACLE + many-body rotation estimator).

Layered so each piece is validated against the next-simpler known answer:

  Layer 1 (this file, first):  FHS link-variable Chern number of occupied Bloch bands -- the
     standard, gauge-invariant single-particle ground truth. Benchmarked on the QWZ model
     (C=-1/0/+1) and a regularized continuum Dirac cone (C=+/-1). This is the oracle the
     rotation-eigenvalue many-body estimator must reproduce (memory lesson M3: benchmark the
     Chern checker on QWZ before trusting any method disagreement).

  Layer 2 (added next):  the C_n rotation operator acting on (positions, spinor) configs, and the
     VMC rotation eigenvalue <R_n> = <psi(R_n^-1 .)/psi(.)>, validated to equal the exact
     single-particle product det<phi_a|R_n|phi_b> for a Slater determinant -- then the
     Fang-Gilbert-Bernevig formula extracts the Chern mod n for the many-body NQS state.

Pure numpy: this is post-processing, not in the VMC hot loop.
"""
import numpy as np

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


# --- benchmark Bloch Hamiltonians -------------------------------------------------------
def qwz_h(kx, ky, M):
    """Qi-Wu-Zhang lattice Chern model. Lower band Chern: 0 (|M|>2), -1 (0<M<2), +1 (-2<M<0)."""
    return np.sin(kx) * SX + np.sin(ky) * SY + (M - np.cos(kx) - np.cos(ky)) * SZ


def dirac_h(kx, ky, M, B=1.0):
    """Regularized continuum Dirac cone: (kx sx + ky sy + (M - B k^2) sz).
    Lower-band Chern = (sign(M)+sign(B))/2 -> +1 for M,B>0; 0 for M<0,B>0 (continuum integer)."""
    return kx * SX + ky * SY + (M - B * (kx * kx + ky * ky)) * SZ


def occupied_grid(h_func, kxs, kys, n_occ=1):
    """Lowest-n_occ eigenvectors of h_func(kx,ky) on the grid kxs x kys.
    Returns u_grid (Nx, Ny, nbands, n_occ) complex."""
    Nx, Ny = len(kxs), len(kys)
    nb = h_func(kxs[0], kys[0]).shape[0]
    u = np.zeros((Nx, Ny, nb, n_occ), dtype=complex)
    for ix, kx in enumerate(kxs):
        for iy, ky in enumerate(kys):
            w, v = np.linalg.eigh(h_func(kx, ky))      # ascending
            u[ix, iy] = v[:, :n_occ]
    return u


# --- Layer 1: FHS link-variable Chern ---------------------------------------------------
def _link(u1, u2):
    """U = det<u1|u2> / |det<u1|u2>| for occupied subspaces u1,u2 (nb, n_occ)."""
    M = u1.conj().T @ u2
    d = np.linalg.det(M)
    return d / np.abs(d)


def chern_fhs(u_grid, periodic=True):
    """Fukui-Hatsugai-Suzuki Chern number from occupied Bloch vectors on a grid.
    u_grid (Nx,Ny,nb,n_occ). For a lattice BZ use periodic=True (wraps Nx->0); for a continuum
    patch use a grid large enough that the Berry curvature vanishes at the edge."""
    Nx, Ny = u_grid.shape[:2]
    total = 0.0
    xr = range(Nx) if periodic else range(Nx - 1)
    yr = range(Ny) if periodic else range(Ny - 1)
    for ix in xr:
        for iy in yr:
            ix1, iy1 = (ix + 1) % Nx, (iy + 1) % Ny
            u00, u10 = u_grid[ix, iy], u_grid[ix1, iy]
            u01, u11 = u_grid[ix, iy1], u_grid[ix1, iy1]
            Ux = _link(u00, u10)
            Uy_x = _link(u10, u11)
            Ux_y = _link(u01, u11)
            Uy = _link(u00, u01)
            total += np.angle(Ux * Uy_x / Ux_y / Uy)
    return total / (2.0 * np.pi)


def chern_qwz(M, Nk=24):
    """Convenience: FHS Chern of the QWZ lower band on an Nk x Nk BZ grid."""
    ks = 2.0 * np.pi * np.arange(Nk) / Nk
    u = occupied_grid(lambda kx, ky: qwz_h(kx, ky, M), ks, ks, n_occ=1)
    return chern_fhs(u, periodic=True)


# --- Layer 2: C_n rotation eigenvalue of the many-body state -----------------------------
# The C_n point operation is a spatial rotation by theta=2pi/n about an axis PERPENDICULAR to the
# 2D plane, accompanied (under SOC) by the spin rotation U_n = exp(-i theta sigma_z/2). Because the
# axis is z, U_n is DIAGONAL: it multiplies an electron of spin s by exp(-i theta (1-2s)/2) -- a pure
# phase, no spin mixing. So for a many-body state,
#     [R_n psi](r_1 s_1, ...) = [ prod_i e^{-i theta (1-2 s_i)/2} ] * psi(R_n^{-1} r_1, s_1; ...)
# and the rotation eigenvalue <R_n> = <psi|R_n|psi>/<psi|psi> is a single VMC expectation value
# (no inter-state overlaps -- the property that makes it NN-VMC friendly; Valenti 2512.07947).

def _rot2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def rotate_positions(R, n, L, center=0.0):
    """R_n^{-1} r = Rot(-2pi/n)(r-center)+center, wrapped to [0,L)^2.  R (...,2)."""
    th = 2.0 * np.pi / n
    Rm = np.asarray(_rot2d(-th))
    c = np.asarray(center) if np.ndim(center) else np.array([center, center], float)
    return np.mod((np.asarray(R) - c) @ Rm.T + c, L)


def spin_rotation_phase(S, n):
    """prod_i exp(-i (2pi/n) (1-2 s_i)/2) over electrons of a config (S (...,N) int)."""
    th = 2.0 * np.pi / n
    return np.exp(-1j * th * np.sum(1.0 - 2.0 * np.asarray(S), axis=-1) / 2.0)


def rotation_eigenvalue_vmc(logpsi_np, R, S, n, L, center=0.0):
    """Estimate <R_n> = < spin_phase * psi(R_n^{-1} R, S)/psi(R,S) >_{|psi|^2} over walker samples.
    logpsi_np(R_one, S_one) -> complex (numpy-callable). R (W,N,2), S (W,N)."""
    Rrot = rotate_positions(R, n, L, center)                  # (W,N,2)
    sp = spin_rotation_phase(S, n)                            # (W,)
    vals = np.array([np.exp(complex(logpsi_np(Rrot[w], S[w])) - complex(logpsi_np(R[w], S[w])))
                     for w in range(R.shape[0])])
    ratios = sp * vals
    return complex(np.mean(ratios)), complex(np.std(ratios) / np.sqrt(len(ratios)))


def rotation_eigenvalue_exact_singleG(Gs, Chi, n, L):
    """Exact <R_n> = det(m) for a Slater determinant of single-plane-wave spinor orbitals
    phi_a = exp(i Gs[a].r) Chi[a] (orthonormal: distinct Gs). Requires the Gs set C_n-symmetric.
        m[a,b] = sum_s Chi[a,s]* e^{-i theta (1-2s)/2} Chi[b,s] * delta(Gs[a], R_n Gs[b]).
    """
    Gs = np.asarray(Gs, float)
    Chi = np.asarray(Chi, complex)
    N = len(Gs)
    th = 2.0 * np.pi / n
    Rp = _rot2d(th)                                           # R_n (forward)
    phase = np.array([np.exp(-1j * th * (1 - 2 * s) / 2.0) for s in (0, 1)])  # s=0,1
    m = np.zeros((N, N), dtype=complex)
    for b in range(N):
        RGb = Rp @ Gs[b]
        for a in range(N):
            if np.allclose(Gs[a], RGb, atol=1e-8):
                m[a, b] = np.sum(np.conj(Chi[a]) * phase * Chi[b])
    return complex(np.linalg.det(m))
