"""Spin-resolved observables for a converged continuum NQS state -- the diagnostics that decide
WHAT a state is, not just its energy.

The verdict "SkX survives / doesn't survive" cannot rest on energy + seed-label alone: pin->release
lets a state relax to anything, and (as in HF) a trivial-seeded run can flow INTO the skyrmion. So we
measure the actual texture and topology of each CONVERGED state:

  m_a(r) = < sum_i sigma_a^i  delta(r - r_i) >        (spin/magnetization density, a in {x,y,z})
  Q      = Berg-Luscher skyrmion number of the unit texture  n(r) = m(r)/|m(r)|
  S(q)   = spin structure factor  (1/N) < |sum_i sigma_a^i e^{-i q.r_i}|^2 >   (ordering wavevector)

Single-particle spin estimators for a sampled config (R, S in {0,1}^N), L = log Psi:
  sigma_z^i = (1 - 2 s_i)                       (DIAGONAL -- just the sampled spin sign)
  sigma_x^i = Psi(s_i->1-s_i)/Psi  = ratio_i    ((sigma_x)_{s,1-s} = 1)
  sigma_y^i = -i (1-2 s_i) ratio_i              ((sigma_y)_{0,1}=-i, (sigma_y)_{1,0}=+i)
i.e. the SAME spin-flip ratio  ratio_i = exp(L(flip i) - L)  used for the SOC local energy. For a
Hermitian operator <O> is real; the per-sample estimator is complex and we take the real part of the
batch mean.

Translation caveat: a symmetry-UNBROKEN finite state has m(r) ~ uniform (the crystal phase averages
out). Measure m(r) with a pin phase-reference (or a short post-release window) for a clean texture; use
the translation-invariant S(q) to detect order regardless of phase. Both are reported.
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def make_spin_estimators(logpsi):
    """Return per_config(R,S) -> (sx, sy, sz) each (N,), the single-particle spin estimators."""
    def per_config(R, S):
        N = R.shape[0]
        L0 = logpsi(R, S)

        def flip_i(i):
            Sf = S.at[i].set(1 - S[i])
            return jnp.exp(logpsi(R, Sf) - L0)                 # ratio_i = Psi(flip i)/Psi, complex
        ratio = jax.vmap(flip_i)(jnp.arange(N))                # (N,) complex
        ratio = jnp.where(jnp.isfinite(ratio), ratio, 0.0 + 0.0j)   # singular flip -> 0
        sz = 1.0 - 2.0 * S.astype(jnp.float64)                 # (N,) real, +-1
        sx = ratio                                             # complex
        sy = -1j * sz * ratio                                  # complex
        return sx, sy, sz
    return per_config


def spin_estimates(logpsi, R_batch, S_batch):
    """Batched spin estimators -> (sx, sy, sz) each (B, N). Real parts are the physical spins."""
    est = jax.jit(jax.vmap(make_spin_estimators(logpsi)))
    sx, sy, sz = est(jnp.asarray(R_batch), jnp.asarray(S_batch))
    return np.array(jnp.real(sx)), np.array(jnp.real(sy)), np.array(jnp.real(sz))


def spin_texture(logpsi, R_batch, S_batch, A, ngrid=12):
    """Real-space magnetization density on an ngrid x ngrid grid over cell A (columns=lattice vecs).
    Returns dict(m (ngrid,ngrid,3), rho (ngrid,ngrid), mbar (3,) net moment/elec, mabs mean |m|)."""
    sx, sy, sz = spin_estimates(logpsi, R_batch, S_batch)       # (B,N) each, real
    B, N = sz.shape
    Ainv = np.linalg.inv(np.asarray(A, float))
    fr = np.mod(np.asarray(R_batch) @ Ainv.T, 1.0)             # (B,N,2) fractional
    gi = np.clip((fr * ngrid).astype(int), 0, ngrid - 1)        # bin indices
    ix, iy = gi[..., 0].ravel(), gi[..., 1].ravel()
    m = np.zeros((ngrid, ngrid, 3))
    rho = np.zeros((ngrid, ngrid))
    for c, sval in enumerate((sx, sy, sz)):
        np.add.at(m[..., c], (ix, iy), sval.ravel())
    np.add.at(rho, (ix, iy), 1.0)
    m /= B                                                     # per-config spin density per bin
    rho /= B
    mbar = m.reshape(-1, 3).sum(0) / N                         # net moment per electron
    mabs = float(np.mean(np.linalg.norm(m, axis=-1)))
    return dict(m=m, rho=rho, mbar=mbar, mabs=mabs)


def berg_luscher(m):
    """Skyrmion number of a texture m (ngrid,ngrid,3) via lattice solid angle (periodic)."""
    ng = m.shape[0]
    n = m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-12)
    tot = 0.0
    for i in range(ng):
        for j in range(ng):
            a = n[i, j]; b = n[(i + 1) % ng, j]
            c = n[i, (j + 1) % ng]; d = n[(i + 1) % ng, (j + 1) % ng]
            for (p, q, w) in [(a, b, d), (a, d, c)]:
                num = np.dot(p, np.cross(q, w))
                den = 1.0 + np.dot(p, q) + np.dot(q, w) + np.dot(w, p)
                tot += 2.0 * np.arctan2(num, den)
    return tot / (4.0 * np.pi)


def structure_factor(logpsi, R_batch, S_batch, A, nshell=4):
    """Spin structure factor S_zz(q) and S_perp(q)=S_xx+S_yy on the reciprocal grid m*b1+n*b2,
    |m|,|n|<=nshell. Returns dict(q (Q,2), Szz (Q,), Sperp (Q,), qfrac list). Translation-invariant."""
    sx, sy, sz = spin_estimates(logpsi, R_batch, S_batch)       # (B,N)
    B, N = sz.shape
    Bmat = 2.0 * np.pi * np.linalg.inv(np.asarray(A, float)).T  # columns b1,b2
    mn = [(p, q) for p in range(-nshell, nshell + 1) for q in range(-nshell, nshell + 1)]
    qs = np.array([p * Bmat[:, 0] + q * Bmat[:, 1] for (p, q) in mn])   # (Q,2)
    R = np.asarray(R_batch)                                     # (B,N,2)
    phase = np.exp(-1j * np.einsum('bnd,qd->bnq', R, qs))       # (B,N,Q)
    rz = np.einsum('bn,bnq->bq', sz, phase)                     # (B,Q)
    rx = np.einsum('bn,bnq->bq', sx, phase)
    ry = np.einsum('bn,bnq->bq', sy, phase)
    Szz = np.mean(np.abs(rz) ** 2, axis=0) / N
    Sperp = np.mean(np.abs(rx) ** 2 + np.abs(ry) ** 2, axis=0) / N
    return dict(q=qs, qfrac=mn, Szz=Szz, Sperp=Sperp)


def spin_order_parameter(logpsi, R_batch, S_batch, A, Gset):
    """Coherent spin order parameter  m_a(G) = (1/N) <sum_i sigma_a^i e^{-iG.r_i}>  for each G in Gset.
    |m(G)|>0 at G!=0 IS broken-symmetry order at wavevector G -- a GLOBAL average, so robust at low
    density where the real-space grid texture is too sparse. Returns (3, nG) complex (a in x,y,z)."""
    sx, sy, sz = spin_estimates(logpsi, R_batch, S_batch)       # (B,N) real
    R = np.asarray(R_batch); N = R.shape[1]
    G = np.asarray(Gset, float)                                 # (nG,2)
    ph = np.exp(-1j * np.einsum('bnd,gd->bng', R, G))           # (B,N,nG)
    return np.array([np.einsum('bn,bng->bg', a, ph).mean(0) / N for a in (sx, sy, sz)])   # (3,nG)


def skx_arms(q1, q2, nharm=1):
    """Skyrmion ordering set {0} U {+-h q_j} for the three arms q1,q2,q3=-(q1+q2), harmonics 1..nharm."""
    q1 = np.asarray(q1, float); q2 = np.asarray(q2, float); q3 = -(q1 + q2)
    Gs = [np.zeros(2)]
    for h in range(1, nharm + 1):
        for q in (q1, q2, q3):
            Gs += [h * q, -h * q]
    return np.array(Gs)


def skyrmion_number_fourier(logpsi, R_batch, S_batch, A, q1, q2, nharm=1, ngrid=24):
    """Robust skyrmion number at low density: measure the coherent m(G) at the skyrmion arms, reconstruct
    the SMOOTH texture m(r)=Re sum_G m(G) e^{iG.r}, and take its Berg-Luscher charge. Also returns the
    order-parameter amplitude (mean |m(G)| over the arms) -- if ~0 the state is disordered and Q is moot."""
    G = skx_arms(q1, q2, nharm)
    mG = spin_order_parameter(logpsi, R_batch, S_batch, A, G)   # (3,nG) complex
    fr = (np.arange(ngrid) + 0.5) / ngrid
    Ucell = np.asarray(A, float)
    m = np.zeros((ngrid, ngrid, 3))
    for i, u in enumerate(fr):
        for j, v in enumerate(fr):
            r = Ucell @ np.array([u, v])
            m[i, j] = np.real(mG @ np.exp(1j * (G @ r)))
    arm_amp = float(np.mean(np.linalg.norm(mG[:, 1:], axis=0)))  # mean |m(G_arm)| over arms
    return dict(Q=float(berg_luscher(m)), order_amp=arm_amp, m_recon=m, mG=mG, m0=np.real(mG[:, 0]))


def slater_spin_density(Gs, W, A, ngrid=24):
    """EXACT spin density m(r)=sum_occ phi_a(r)^dagger sigma phi_a(r) of the Slater determinant with
    orbitals phi_a(r,s)=sum_G W[a,G,s] e^{iG.r} -- no MCMC. Use to read the skyrmion number of a
    warm-start determinant (Gs,W) directly. Returns m (ngrid,ngrid,3) real."""
    Gs = np.asarray(Gs, float); W = np.asarray(W, complex); Ac = np.asarray(A, float)
    fr = (np.arange(ngrid) + 0.5) / ngrid
    m = np.zeros((ngrid, ngrid, 3))
    for i, u in enumerate(fr):
        for j, v in enumerate(fr):
            r = Ac @ np.array([u, v])
            phi = (W * np.exp(1j * (Gs @ r))[None, :, None]).sum(1)    # (N,2) occupied spinors at r
            up, dn = phi[:, 0], phi[:, 1]
            cud = np.sum(np.conj(up) * dn)
            m[i, j] = [2 * np.real(cud), 2 * np.imag(cud), float(np.sum(np.abs(up) ** 2 - np.abs(dn) ** 2))]
    return m


def classify_state(logpsi, R_batch, S_batch, A, ngrid=12, nshell=4, q1=None, q2=None, nharm=1):
    """One-call diagnostic: texture + skyrmion number + structure-factor peaks + a heuristic label.
    If the ordering arms (q1,q2) are given, the skyrmion number comes from the ROBUST Fourier
    reconstruction (skyrmion_number_fourier) and `order_amp` reports how ordered the state is."""
    tex = spin_texture(logpsi, R_batch, S_batch, A, ngrid=ngrid)
    if q1 is not None:
        fou = skyrmion_number_fourier(logpsi, R_batch, S_batch, A, q1, q2, nharm=nharm, ngrid=max(20, ngrid))
        Q = fou["Q"]; order_amp = fou["order_amp"]
    else:
        Q = berg_luscher(tex["m"]); order_amp = float("nan"); fou = None
    sf = structure_factor(logpsi, R_batch, S_batch, A, nshell=nshell)
    # dominant finite-q peak (exclude q=0)
    qn = np.linalg.norm(sf["q"], axis=1)
    fin = qn > 1e-9
    tot = sf["Szz"] + sf["Sperp"]
    gamma = float(tot[~fin].sum())                              # q=0 weight (FM-like)
    if fin.any():
        kpk = np.argmax(tot * fin)
        peak_q = sf["q"][kpk]; peak_frac = sf["qfrac"][kpk]; peak_val = float(tot[kpk])
        zfrac = float(sf["Szz"][kpk] / (tot[kpk] + 1e-12))     # is the finite-q peak z (SDW) or in-plane?
    else:
        peak_q = np.zeros(2); peak_frac = (0, 0); peak_val = 0.0; zfrac = 0.0
    mz, mxy = abs(tex["mbar"][2]), float(np.hypot(*tex["mbar"][:2]))
    finite_q_order = peak_val > 1.5 * (gamma / max(1, (~fin).sum()))
    # "ordered" = a real broken-symmetry order parameter (robust Fourier amp), else fall back to S(q)
    ordered = (order_amp > 0.05) if order_amp == order_amp else finite_q_order
    if not ordered:
        label = ("ferromagnet (m_z)" if mz > 0.3 else
                 "in-plane / Rashba-FM" if mxy > 0.3 else
                 "fluid / disordered")
    else:
        label = ("SKYRMION (Q!=0, ordered)" if abs(Q) > 0.4 else
                 "z-SDW / collinear (Overhauser)" if zfrac > 0.6 else
                 "spiral / in-plane (1Q)")
    return dict(skyrmion_Q=float(Q), order_amp=order_amp, mbar=tex["mbar"].tolist(), mabs=tex["mabs"],
                peak_qfrac=peak_frac, peak_val=peak_val, gamma_weight=gamma,
                peak_is_z=zfrac, label=label, _tex=tex, _sf=sf, _fou=fou)
