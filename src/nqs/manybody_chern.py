"""Many-body Chern number via twisted boundary conditions (Niu-Thouless-Wu) for the NQS.

The correlated state |psi(theta)> is trained/relaxed on a grid of boundary twists theta in [0,2pi)^2.
The many-body Chern number is the FHS lattice field strength of the U(1) links
    U_mu(theta) = <psi(theta)|psi(theta+mu)> / |<psi(theta)|psi(theta+mu)>|
summed over the twist torus:  C = (1/2pi) sum_p Arg(U1 U2 U3^* U4^*).
sigma_xy = C e^2/h. Gauge-safe: independent per-twist global phases/normalizations cancel in
plaquettes (each state enters once as bra and once as ket around every loop).

Overlap estimator (one-sided; neighbors on a fine grid are close so the variance is benign):
sampling R,S ~ |psi_a|^2 and r = exp(logpsi_b - logpsi_a),
    O_ab = <psi_a|psi_b>/sqrt(<a|a><b|b>) = E_a[r] / sqrt(E_a[|r|^2]).
|O| well below ~0.5 on any link flags a too-coarse grid or a branch jump -- REFINE, don't trust.
Validation (tests/test_manybody_chern.py): the estimator reproduces the EXACT determinant overlap
det(Wa^dag Wb) for plane-wave Slater pairs; the FHS assembly reproduces C(QWZ) = -1/0 exactly and is
gauge-invariant under random per-point phases.
"""
import numpy as np
import jax
import jax.numpy as jnp


def mc_overlap(logpsi_a, logpsi_b, R, S, batch=4096):
    """Normalized overlap O_ab from samples (R,S) drawn from |psi_a|^2. Returns (O complex, |O|, ess)."""
    la = jax.jit(jax.vmap(logpsi_a))
    lb = jax.jit(jax.vmap(logpsi_b))
    d = np.asarray(jax.device_get(lb(R, S) - la(R, S)))
    d = d - d.real.max()                                    # overflow guard (cancels in the ratio)
    r = np.exp(d)
    num = r.mean()
    den = np.sqrt((np.abs(r) ** 2).mean())
    ess = float((np.abs(r).sum() ** 2) / (np.abs(r) ** 2).sum() / len(r))   # effective-sample fraction
    O = num / den
    return complex(O), float(abs(O)), ess


def fhs_chern(links_x, links_y):
    """FHS Chern number from link matrices on an (nt x nt) periodic twist grid.
    links_x[i,j] = U_x(theta_ij -> theta_{i+1,j}), links_y[i,j] = U_y(theta_ij -> theta_{i,j+1}).
    Returns (C float, F (nt,nt) plaquette fluxes in (-pi,pi])."""
    lx, ly = np.asarray(links_x), np.asarray(links_y)
    nt = lx.shape[0]
    F = np.zeros((nt, nt))
    for i in range(nt):
        for j in range(nt):
            u = (lx[i, j] * ly[(i + 1) % nt, j] * np.conj(lx[i, (j + 1) % nt]) * np.conj(ly[i, j]))
            F[i, j] = np.angle(u)
    return float(F.sum() / (2 * np.pi)), F


def chern_report(links_x, links_y, absx, absy):
    """C + diagnostics: min |O| (grid adequacy), max |F| (flux concentration; near pi = suspect)."""
    C, F = fhs_chern(links_x, links_y)
    return dict(C=C, C_int=int(np.rint(C)), F=F.tolist(),
                minO=float(min(np.min(absx), np.min(absy))),
                maxF_over_pi=float(np.max(np.abs(F)) / np.pi))
