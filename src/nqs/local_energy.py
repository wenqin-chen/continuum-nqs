"""Local energy of a spinful continuum wavefunction with Rashba SOC (JAX, generic).

E_loc(R,S) = Psi^{-1} H Psi  for the single-particle-sum Hamiltonian

    H = sum_i [ -1/(2 m*) grad_i^2  +  lambda_R (sigma_x^i (-i d_{y_i}) + sigma_y^i (-i d_{x_i}))
                -  (h_z/2) sigma_z^i ]   ( + Coulomb, added separately later )

matching the reference Hartree-Fock engine EXACTLY (SOC = sigma_x k_y + sigma_y k_x).

Spin is sampled (S in {0,1}^N). Writing L = log Psi (complex):
  * kinetic   : -1/(2m*) sum_i ( grad_i^2 L + (grad_i L).(grad_i L) )      [diagonal in spin]
  * Zeeman    : -(h_z/2) sum_i (sigma_z)_{s_i s_i} = -(h_z/2) sum_i (1-2 s_i)  [diagonal]
  * SOC       : off-diagonal in spin -> for each i, flip s_i and use the ratio + position
                gradient of the flipped amplitude:
                E_SOC,i = lambda_R sum_{s'} [ (sigma_x)_{s_i s'} (-i) d_{y_i} Psi(s')
                                            + (sigma_y)_{s_i s'} (-i) d_{x_i} Psi(s') ] / Psi
                only s' = 1 - s_i contributes (sigma_x, sigma_y are off-diagonal).

The function is GENERIC in `logpsi` (R,S)->complex, so the same operator validates the
exact Slater trial today and scores the Psiformer later. Complex autodiff is done by
differentiating Re/Im separately (both real-valued), avoiding holomorphic ambiguities.
Kinetic uses a dense Hessian here (O((2N)^2) AD; fine for N<=~16); swap in Forward-Laplacian
for production.
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

try:
    from folx import forward_laplacian       # Forward-Laplacian (O(N) kinetic) for production/GPU
    _HAS_FOLX = True
except ImportError:
    _HAS_FOLX = False

_SX = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
_SY = jnp.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=jnp.complex128)


def make_local_energy(logpsi, m_star=1.0, lambda_r=1.0, h_z=0.0, ext_field=None, use_folx=False):
    """Return E_loc(R, S) -> complex scalar for a fixed logpsi(R,S).

    ext_field(r) -> (3,) real adds an external spin coupling  -sum_i ext_field(r_i) . sigma_i
    (e.g. a spin spiral B*(cos Q.r, sin Q.r, 0)); None = pure kinetic + SOC + Zeeman.

    use_folx=True computes the kinetic Laplacian via the folx Forward-Laplacian (a single forward
    pass, ~O(N) -- for N~20-30 production) instead of the dense Hessian (O((2N)^2), the default and
    the validation reference)."""
    if use_folx and not _HAS_FOLX:
        raise RuntimeError("use_folx=True but folx is not installed (pip install folx)")

    def _Lr(rf, S, N):
        return jnp.real(logpsi(rf.reshape(N, 2), S))

    def _Li(rf, S, N):
        return jnp.imag(logpsi(rf.reshape(N, 2), S))

    gLr = jax.grad(_Lr, argnums=0)
    gLi = jax.grad(_Li, argnums=0)
    hLr = jax.hessian(_Lr, argnums=0)
    hLi = jax.hessian(_Li, argnums=0)

    def E_loc(R, S):
        N = R.shape[0]
        rf = R.reshape(-1)
        L0 = logpsi(R, S)

        # --- kinetic: grad L and Laplacian L (folx Forward-Laplacian, or dense Hessian) ---
        if use_folx:
            def _reim(rf_):
                v = logpsi(rf_.reshape(N, 2), S)
                return jnp.stack([jnp.real(v), jnp.imag(v)])
            o = forward_laplacian(_reim)(rf)
            jac = o.jacobian.dense_array                  # (2N, 2): [d Re/d rf, d Im/d rf]
            g = jac[:, 0] + 1j * jac[:, 1]                # (2N,) complex grad of L
            lap = o.laplacian[0] + 1j * o.laplacian[1]    # complex Laplacian of L
        else:
            g = gLr(rf, S, N) + 1j * gLi(rf, S, N)        # (2N,) complex grad of L
            lap = jnp.trace(hLr(rf, S, N)) + 1j * jnp.trace(hLi(rf, S, N))
        grad2 = jnp.sum(g * g)                            # (grad L).(grad L), complex
        E_kin = -1.0 / (2.0 * m_star) * (lap + grad2)

        # --- Zeeman + ext-field z component (both diagonal in spin) ---
        sgn = 1.0 - 2.0 * S.astype(jnp.float64)           # (sigma_z)_{s s} = +1 (s=0), -1 (s=1)
        E_Z = -0.5 * h_z * jnp.sum(sgn)
        if ext_field is not None:
            fz = jax.vmap(lambda r: ext_field(r)[2])(R)   # (N,)
            E_Z = E_Z - jnp.sum(fz * sgn)                 # -sum_i fz_i (sigma_z)_{s_i s_i}

        # --- SOC (spin-off-diagonal): vmap over electrons ---
        def soc_i(i):
            s_i = S[i]
            s_p = 1 - s_i
            Sflip = S.at[i].set(s_p)
            Lflip = logpsi(R, Sflip)
            ratio = jnp.exp(Lflip - L0)                   # Psi(s')/Psi(s)
            gflip = gLr(rf, Sflip, N) + 1j * gLi(rf, Sflip, N)
            # d_{x_i,y_i} Psi(s') / Psi(s) = ratio * d_{x_i,y_i} L(s').
            # If the flipped determinant is exactly singular (e.g. a fully spin-polarized
            # state), Psi(s')=0 so the true coupling is 0, but numerically ratio=0 and
            # the singular logdet gradient is inf -> 0*inf=NaN; guard it back to 0.
            ampgx = ratio * gflip[2 * i]
            ampgy = ratio * gflip[2 * i + 1]
            ampgx = jnp.where(jnp.isfinite(ampgx), ampgx, 0.0 + 0.0j)
            ampgy = jnp.where(jnp.isfinite(ampgy), ampgy, 0.0 + 0.0j)
            sx = _SX[s_i, s_p]
            sy = _SY[s_i, s_p]
            e = lambda_r * (sx * (-1j) * ampgy + sy * (-1j) * ampgx)
            if ext_field is not None:                     # -[fx sigma_x + fy sigma_y]_{s_i,s'} * ratio
                f = ext_field(R[i])
                e = e - (f[0] * sx + f[1] * sy) * ratio
            return e

        E_SOC = jnp.sum(jax.vmap(soc_i)(jnp.arange(N)))
        return E_kin + E_Z + E_SOC

    return E_loc
