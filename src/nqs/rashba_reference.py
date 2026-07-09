"""Analytic non-interacting periodic Rashba 2DEG -- an exact validation oracle.

Single-particle Hamiltonian, EXACTLY matching the reference Hartree-Fock engine (
diag_blocks):

    h0(k) = (kx^2 + ky^2)/(2 m*) I2  +  lambda_R (sigma_x k_y + sigma_y k_x)  -  (h_z/2) sigma_z

NOTE the SOC convention is (sigma_x k_y + sigma_y k_x), NOT the textbook
(p_x sigma_y - p_y sigma_x).  In real space H_SOC = lambda_R (sigma_x p_y + sigma_y p_x),
p = -i grad.  Get this wrong and the NQS is not solving the same problem as the HF.

Writing the d-vector h0 = k^2/(2m*) I + d.sigma with d = (lambda_R k_y, lambda_R k_x, -h_z/2):

    |d| = sqrt(lambda_R^2 (kx^2+ky^2) + h_z^2/4)
    E_pm(k) = k^2/(2 m*) +/- |d|                              (lower band: minus)

For h_z -> 0 the lower band E_-(k) = k^2/2m* - lambda_R |k| has a ring minimum at
|k| = lambda_R m*, depth E_min = -lambda_R^2 m*/2 (the Rashba ring).

The non-interacting N-electron ground state is a SINGLE Slater determinant of the lowest N
spin-orbitals  phi_a(r,s) = exp(i k_a . r) chi_a(s) / sqrt(A),  chi_a = the band spinor.
This module provides:
  (i)  the EXACT GS energy, to validate the NQS against (G1 gate);
  (ii) the occupied plane-wave spinors, to warm-start the generalized-determinant envelopes.

Pure numpy (no JAX): this is a reference oracle, not part of the sampled wavefunction.
"""
import numpy as np

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def h0_k(kx, ky, m_star=1.0, lambda_r=1.0, h_z=0.0):
    """The 2x2 single-particle Bloch Hamiltonian at momentum (kx, ky)."""
    kin = (kx * kx + ky * ky) / (2.0 * m_star)
    return kin * I2 + lambda_r * (SX * ky + SY * kx) - 0.5 * h_z * SZ


def bands_k(kx, ky, m_star=1.0, lambda_r=1.0, h_z=0.0):
    """(E_lower, E_upper, chi_lower, chi_upper).  Energies analytic; spinors via eigh
    (robust at k=0, where d=(0,0,-h_z/2))."""
    w, v = np.linalg.eigh(h0_k(kx, ky, m_star, lambda_r, h_z))  # ascending
    return float(w[0]), float(w[1]), v[:, 0].copy(), v[:, 1].copy()


def bands_analytic(kx, ky, m_star=1.0, lambda_r=1.0, h_z=0.0):
    """Closed-form band energies (no diagonalization) -- for cross-checking eigh."""
    k2 = kx * kx + ky * ky
    kin = k2 / (2.0 * m_star)
    dmag = np.sqrt(lambda_r * lambda_r * k2 + 0.25 * h_z * h_z)
    return kin - dmag, kin + dmag


def allowed_k(L, n_cut, twist=(0.0, 0.0)):
    """Momenta on the L x L torus with Bloch twist `twist` in [0,2pi)^2:
        k = (2 pi n + twist) / L,   n in Z^2,  |n_x|,|n_y| <= n_cut.
    Returns flattened (kx, ky)."""
    j = np.arange(-n_cut, n_cut + 1)
    JX, JY = np.meshgrid(j, j, indexing="ij")
    kx = (2.0 * np.pi * JX + twist[0]) / L
    ky = (2.0 * np.pi * JY + twist[1]) / L
    return kx.ravel(), ky.ravel()


def single_particle_levels(L, m_star=1.0, lambda_r=1.0, h_z=0.0, n_cut=8, twist=(0.0, 0.0)):
    """All 2*(2 n_cut+1)^2 spin-orbital levels, sorted ascending in energy.
    Each entry: dict(E, kx, ky, band, n (integer index), chi (2-spinor))."""
    kx, ky = allowed_k(L, n_cut, twist)
    j = np.arange(-n_cut, n_cut + 1)
    JX, JY = [a.ravel() for a in np.meshgrid(j, j, indexing="ij")]
    levels = []
    for x, y, nx, ny in zip(kx, ky, JX, JY):
        El, Eu, cl, cu = bands_k(x, y, m_star, lambda_r, h_z)
        levels.append(dict(E=El, kx=x, ky=y, band=0, n=(int(nx), int(ny)), chi=cl))
        levels.append(dict(E=Eu, kx=x, ky=y, band=1, n=(int(nx), int(ny)), chi=cu))
    levels.sort(key=lambda d: d["E"])
    return levels


def noninteracting_gs(N, L, m_star=1.0, lambda_r=1.0, h_z=0.0, n_cut=8, twist=(0.0, 0.0)):
    """Exact non-interacting N-electron ground state on the torus.

    Returns dict with:
      E_gs      total ground-state energy (sum of lowest N levels)
      E_per     energy per electron
      gap       E_LUMO - E_HOMO (>0  => closed shell, unique single-determinant GS)
      occ       the N occupied levels (each a dict from single_particle_levels)
      density   N / L^2
      cutoff_ok True if the highest occupied |k|^2 is well inside the n_cut grid edge
    """
    levels = single_particle_levels(L, m_star, lambda_r, h_z, n_cut, twist)
    if len(levels) < N + 1:
        raise ValueError("n_cut too small for N levels")
    occ = levels[:N]
    E_gs = float(sum(d["E"] for d in occ))
    gap = float(levels[N]["E"] - levels[N - 1]["E"])
    k2_occ = max(d["kx"] ** 2 + d["ky"] ** 2 for d in occ)
    k2_edge = (2.0 * np.pi * n_cut / L) ** 2
    return dict(
        E_gs=E_gs, E_per=E_gs / N, gap=gap, occ=occ,
        E_homo=float(occ[-1]["E"]), E_lumo=float(levels[N]["E"]),
        density=N / (L * L), N=N, L=L, m_star=m_star, lambda_r=lambda_r, h_z=h_z,
        twist=twist, k2_occ_max=k2_occ, k2_cut=k2_edge,
        cutoff_ok=bool(k2_occ < 0.5 * k2_edge),
    )


def occupied_orbital_matrix(gs):
    """Stack the occupied spin-orbitals as the warm-start envelope data.
    Returns (Kvecs (N,2), Chi (N,2) complex): plane-wave momenta and band spinors, so that
        phi_a(r, s) = exp(i Kvecs[a] . r) * Chi[a, s] / sqrt(L^2).
    This is the exact non-interacting GS determinant and the seed for the NQS envelopes."""
    occ = gs["occ"]
    Kvecs = np.array([[d["kx"], d["ky"]] for d in occ], dtype=float)
    Chi = np.array([d["chi"] for d in occ], dtype=complex)
    return Kvecs, Chi


def planewave_envelope(Kvecs, Chi):
    """Convert single-plane-wave orbitals (Kvecs (n,2), Chi (n,2)) to the general multi-G
    envelope (Gs (n,2), W (n,n,2)) used by the Psiformer / make_logpsi_general:
        phi_a(r,s) = sum_g W[a,g,s] exp(i Gs[g].r),  W[a,g,:] = Chi[a]*delta(g,a).
    The uniform Rashba GS is the special case one plane wave per orbital."""
    Kvecs = np.asarray(Kvecs, dtype=float)
    Chi = np.asarray(Chi, dtype=complex)
    n = Kvecs.shape[0]
    W = np.zeros((n, n, 2), dtype=complex)
    for a in range(n):
        W[a, a, :] = Chi[a]
    return Kvecs.copy(), W


def cell_rashba_envelope(A, N, m_star=1.0, lambda_r=1.0, h_z=0.0, n_cut=8, twist=(0.0, 0.0)):
    """Occupied non-interacting Rashba orbitals on a GENERAL cell A (columns = lattice vectors) ->
    the base warm-start envelope (Gs (N,2), W (N,N,2)) + E_gs. This is the uniform-liquid base for
    the target supercell; the SkX is then nucleated by a pinning field (ext_field). Allowed momenta
    k = B n + (A^-1)^T twist, B = 2pi (A^-1)^T. A may be a scalar L (square box)."""
    A = (np.array([[float(A), 0.0], [0.0, float(A)]]) if np.ndim(A) == 0 else np.asarray(A, float))
    Ainv = np.linalg.inv(A)
    B = 2.0 * np.pi * Ainv.T
    kt = Ainv.T @ np.asarray(twist, float)
    js = np.arange(-n_cut, n_cut + 1)
    levels = []
    for nx in js:
        for ny in js:
            k = B @ np.array([float(nx), float(ny)]) + kt
            El, Eu, cl, cu = bands_k(k[0], k[1], m_star, lambda_r, h_z)
            levels.append((El, k, cl))
            levels.append((Eu, k, cu))
    levels.sort(key=lambda t: t[0])
    occ = levels[:N]
    Kvecs = np.array([t[1] for t in occ])
    Chi = np.array([t[2] for t in occ])
    Gs, W = planewave_envelope(Kvecs, Chi)
    return dict(Gs=Gs, W=W, E_gs=float(sum(t[0] for t in occ)),
                gap=float(levels[N][0] - levels[N - 1][0]), Kvecs=Kvecs, Chi=Chi,
                area=float(abs(np.linalg.det(A))), N=N, density=N / float(abs(np.linalg.det(A))))


def cell_skx_envelope(A, N, s_keys, s_vals, q1, q2, lambda_tex=1.0,
                      m_star=1.0, lambda_r=1.0, h_z=0.0, n_cut=8, twist=(0.0, 0.0), gtol=1e-3,
                      n_keys=None, n_vals=None, lambda_chg=0.0):
    """SKYRMION-BACKGROUND determinant: occupied orbitals of  h0(k) + V_texture  on the supercell A,
    where V(r) = -lambda_tex * n(r).sigma is the HF spin texture n(r)=Re sum_Q s_vals[Q] e^{iQ.r} as a
    STATIC potential (Q = m q1 + n q2 from s_keys). The texture couples plane wave k -> k+Q (off-diagonal
    in G with spin matrix -lambda_tex * sum_a s_vals[Q,a] sigma_a), so the occupied eigen-orbitals are
    MULTI-G spinors whose spin density carries the texture's winding (skyrmion number != 0). This is the
    HF-orbital warm-start the pinning protocol could not impose: feed (Gs,W) to make_logpsi_general for an
    exact skyrmion determinant (zero-variance under h0+V), or to the Psiformer envelope to optimize from it.
    lambda_tex tunes how strongly the orbitals follow the texture -- scan it so the measured Q matches the HF.
    Returns dict(Gs (npw,2), W (N,npw,2), E_band, gap, w, npw, lambda_tex)."""
    A = (np.array([[float(A), 0.0], [0.0, float(A)]]) if np.ndim(A) == 0 else np.asarray(A, float))
    Ainv = np.linalg.inv(A); B = 2.0 * np.pi * Ainv.T; kt = Ainv.T @ np.asarray(twist, float)
    js = np.arange(-n_cut, n_cut + 1)
    Ks = np.array([B @ np.array([float(nx), float(ny)]) + kt for nx in js for ny in js])  # (npw,2)
    npw = len(Ks)
    keyf = lambda k: (round(float(k[0]), 6), round(float(k[1]), 6))
    idx = {keyf(Ks[i]): i for i in range(npw)}
    q1 = np.asarray(q1, float); q2 = np.asarray(q2, float)
    Qs = np.array([m * q1 + n * q2 for (m, n) in np.asarray(s_keys)])   # (nQ,2)
    Sv = np.asarray(s_vals, complex)                                    # (nQ,3)
    sig = [SX, SY, SZ]
    H = np.zeros((2 * npw, 2 * npw), complex)
    for i in range(npw):
        H[2 * i:2 * i + 2, 2 * i:2 * i + 2] = h0_k(Ks[i, 0], Ks[i, 1], m_star, lambda_r, h_z)
    for iq in range(len(Qs)):
        MQ = -lambda_tex * sum(Sv[iq, a] * sig[a] for a in range(3))    # 2x2 spin matrix at Q
        for i in range(npw):
            j = idx.get(keyf(Ks[i] + Qs[iq]))
            if j is not None:
                H[2 * j:2 * j + 2, 2 * i:2 * i + 2] += MQ               # <k+Q,s'|V|k,s>
    if n_keys is not None and lambda_chg != 0.0:                        # HARTREE charge modulation (spin-indep)
        Qn = np.array([m * q1 + n * q2 for (m, n) in np.asarray(n_keys)])
        Nv = np.asarray(n_vals, complex); I2 = np.eye(2)
        for iq in range(len(Qn)):
            Qmag = float(np.hypot(Qn[iq, 0], Qn[iq, 1]))
            if Qmag < 1e-9:
                continue                                                # skip uniform G=0 (constant shift)
            MQc = lambda_chg * (2.0 * np.pi / Qmag) * Nv[iq] * I2        # lambda_chg * V(Q) n_Q, identity in spin
            for i in range(npw):
                j = idx.get(keyf(Ks[i] + Qn[iq]))
                if j is not None:
                    H[2 * j:2 * j + 2, 2 * i:2 * i + 2] += MQc
    H = 0.5 * (H + H.conj().T)                                          # Hermitize (safety)
    w, v = np.linalg.eigh(H)
    occ = v[:, :N]                                                     # (2npw, N)
    W = np.transpose(occ.reshape(npw, 2, N), (2, 0, 1))                # (N, npw, 2): W[a,g,s]
    Gs = Ks.copy()
    # SPARSIFY: each skyrmion orbital is a few plane waves (k0 + texture harmonics); drop the
    # negligible ones so the multi-G envelope doesn't blow up memory in the local energy/measurement
    # (W[:,:,S] vmapped over electrons x walkers scales with n_G).
    wt = np.sum(np.abs(W) ** 2, axis=(0, 2))                           # (npw,) total occupied weight per G
    keep = wt > gtol * wt.max()
    Gs, W = Gs[keep], W[:, keep, :]
    return dict(Gs=Gs, W=W, E_band=float(np.sum(w[:N])), gap=float(w[N] - w[N - 1]),
                w=w, npw=npw, n_G=int(keep.sum()), lambda_tex=lambda_tex)


def ring_minimum(m_star=1.0, lambda_r=1.0, h_z=0.0):
    """Continuum lower-band minimum (k_R, E_min). For h_z=0: k_R=lambda_R m*,
    E_min = -lambda_R^2 m*/2. For h_z>0 the minimum shifts; solve d/dk[k^2/2m* - |d|]=0."""
    if h_z == 0.0:
        return lambda_r * m_star, -0.5 * lambda_r ** 2 * m_star
    ks = np.linspace(0.0, 3.0 * lambda_r * m_star + 1.0, 200001)
    E = ks ** 2 / (2 * m_star) - np.sqrt(lambda_r ** 2 * ks ** 2 + 0.25 * h_z ** 2)
    i = int(np.argmin(E))
    return float(ks[i]), float(E[i])


if __name__ == "__main__":
    # quick human-readable sanity dump at a generic gapped cell
    g = noninteracting_gs(N=7, L=8.0, lambda_r=1.0, h_z=0.5, n_cut=8, twist=(0.1, 0.17))
    kR, Emin = ring_minimum(lambda_r=1.0, h_z=0.5)
    print(f"GS(N=7,L=8,lr=1,hz=0.5): E_gs={g['E_gs']:+.6f}  E_per={g['E_per']:+.6f}  "
          f"gap={g['gap']:.4f}  cutoff_ok={g['cutoff_ok']}  n_dens={g['density']:.4f}")
    print(f"Rashba ring: k_R={kR:.4f}  E_min={Emin:+.4f}")
