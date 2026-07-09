"""Spinor Slater determinant trial -- exact non-interacting wavefunctions (JAX).

GENERAL (multi-G) form: each orbital a is a plane-wave-spinor expansion
    phi_a(r, s) = sum_G  W[a, G, s] * exp(i Gs[G] . r)
    Psi(R, S)   = det[ M ],   M[a, i] = phi_a(r_i, s_i)

This subsumes (i) the uniform Rashba GS (one plane wave per orbital), (ii) a non-collinear
spin spiral (each orbital a comb {k0+nQ}), and (iii) any HF texture (e.g. a skyrmion-crystal warm start) --
all are just different (Gs, W). For occupied EIGEN-orbitals of a non-interacting H, this Psi is the
exact ground state, so E_loc is constant (zero variance) -- the certificate used throughout.

Spin is SAMPLED (s_i in {0,1}); the 1/sqrt(A)^N normalization is a constant prefactor (drops out
of E_loc and MCMC ratios) and is omitted.
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def make_logpsi_general(Gs, W):
    """logpsi(R, S) for the general multi-G spinor determinant.
      Gs : (n_G, 2)   real plane-wave momenta
      W  : (n_orb, n_G, 2) complex spinor coefficients, phi_a(r,s)=sum_G W[a,G,s] exp(i Gs[G].r)
    """
    Gs = jnp.asarray(Gs, dtype=jnp.float64)
    W = jnp.asarray(W, dtype=jnp.complex128)

    def logpsi(R, S):
        phase = jnp.exp(1j * (Gs @ R.T))          # (n_G, N): exp(i G . r_i)
        Ws = W[:, :, S]                            # (n_orb, n_G, N): W[a, G, s_i]
        M = jnp.einsum("gi,agi->ai", phase, Ws)    # (n_orb, N): phi_a(r_i, s_i)
        sign, logabs = jnp.linalg.slogdet(M)
        return logabs + jnp.log(sign)
    return logpsi


def make_logpsi_slater(Kvecs, Chi):
    """Single-plane-wave-per-orbital special case (the uniform Rashba GS):
    phi_a(r,s) = exp(i K_a . r) chi_a(s).  Kept for the uniform-cell tests."""
    Kvecs = jnp.asarray(Kvecs, dtype=jnp.float64)
    Chi = jnp.asarray(Chi, dtype=jnp.complex128)
    n = Kvecs.shape[0]
    W = jnp.zeros((n, n, 2), dtype=jnp.complex128).at[jnp.arange(n), jnp.arange(n), :].set(Chi)
    return make_logpsi_general(Kvecs, W)
