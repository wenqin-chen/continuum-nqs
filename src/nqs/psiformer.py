"""Continuum spinful Psiformer (Flax).

A periodic, spinful Psiformer with a single generalized-spinor determinant
(Avdoshkin-Geier-Fu arXiv:2510.18621 given Psiformer-strength attention + periodic-FermiNet
input features). For SAMPLED spin (s_i in {0,1}) the determinant is N x N:

    M[a, i] = env_a(r_i)[s_i] * (1 + bf[a, i]),     env_a(r)[s] = exp(i K_a . r) * Chi[a, s]
    log Psi(R, S) = logdet M     ( + log-sum-exp over n_dets )

  * env_a : Bloch-plane-wave spinor envelope, warm-started from the HF / Rashba occupied
            orbitals (K_a, Chi_a). Frozen by default (learnable_env=True to relax).
  * bf    : complex backflow from a pre-norm multi-head self-attention backbone over the
            electrons (tokens = electrons; periodic position features + spin). ZERO-INITIALISED,
            so at init bf=0 and M is EXACTLY the validated plane-wave Slater trial
            (logPsi = exact non-interacting GS, E_loc = E_gs to machine precision).

Antisymmetry: swapping (r_i,s_i)<->(r_j,s_j) swaps determinant columns -> sign flip. The block
"full determinant" is non-antisymmetric under SOC (arXiv:2506.00155); this single generalized
determinant is the valid construction.

Backbone real (float64); envelopes + backflow complex (complex128) for the SOC phase structure.
Translation-COVARIANT periodic features (the crystal breaks translation; the envelope carries the
absolute phase). Forward pass is single-configuration (R (N,2), S (N,)); vmap for batches.
"""
from __future__ import annotations
from typing import Any
import flax.linen as fnn
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from .cell import square_cell


# --- attention internals (ported from neural_nets_101/transformer_nqs.py) ---------------
def split_heads(x, n_heads):
    *lead, N, d = x.shape
    hd = d // n_heads
    return jnp.moveaxis(x.reshape(*lead, N, n_heads, hd), -2, -3)


def merge_heads(x):
    x = jnp.moveaxis(x, -3, -2)
    *lead, N, nh, hd = x.shape
    return x.reshape(*lead, N, nh * hd)


class MultiHeadAttention(fnn.Module):
    embed_dim: int
    n_heads: int
    param_dtype: Any = jnp.float64

    @fnn.compact
    def __call__(self, x):
        d = self.embed_dim
        Q = split_heads(fnn.Dense(d, param_dtype=self.param_dtype, name="q")(x), self.n_heads)
        K = split_heads(fnn.Dense(d, param_dtype=self.param_dtype, name="k")(x), self.n_heads)
        V = split_heads(fnn.Dense(d, param_dtype=self.param_dtype, name="v")(x), self.n_heads)
        scale = 1.0 / jnp.sqrt(d // self.n_heads)
        attn = jax.nn.softmax(jnp.einsum("...hid,...hjd->...hij", Q, K) * scale, axis=-1)
        out = merge_heads(jnp.einsum("...hij,...hjd->...hid", attn, V))
        return fnn.Dense(d, param_dtype=self.param_dtype, name="o")(out)


class TransformerBlock(fnn.Module):
    embed_dim: int
    n_heads: int
    ff_mult: int = 2
    param_dtype: Any = jnp.float64

    @fnn.compact
    def __call__(self, x):
        x = x + MultiHeadAttention(self.embed_dim, self.n_heads, self.param_dtype,
                                   name="attn")(fnn.LayerNorm(param_dtype=self.param_dtype)(x))
        h = fnn.LayerNorm(param_dtype=self.param_dtype)(x)
        h = fnn.Dense(self.ff_mult * self.embed_dim, param_dtype=self.param_dtype)(h)
        h = jax.nn.gelu(h)
        h = fnn.Dense(self.embed_dim, param_dtype=self.param_dtype)(h)
        return x + h


def periodic_features(R, cell, n_freq):
    """Translation-covariant periodic features via FRACTIONAL coords s in [0,1)^2 of the cell:
    [sin(2pi m s), cos(2pi m s)] for m=1..n_freq, each axis. cell = scalar L (square) or 2x2 A.
    R (N,2) -> (N, 4*n_freq)."""
    A = jnp.asarray(square_cell(cell) if np.ndim(cell) == 0 else np.asarray(cell, float))
    # fractional coords WITHOUT mod: sin/cos(2pi m s) are already lattice-periodic (s->s+integer
    # leaves them invariant), and dropping jnp.mod keeps folx's Forward-Laplacian fast (mod is not
    # in its registry -> it would silently fall back to the dense Hessian).
    s = R @ jnp.linalg.inv(A).T                           # (N,2)
    ph = 2.0 * jnp.pi * s                                  # (N,2)
    m = jnp.arange(1, n_freq + 1)                          # (n_freq,)
    ang = ph[:, :, None] * m[None, None, :]               # (N,2,n_freq)
    feats = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)  # (N,2,2 n_freq)
    return feats.reshape(R.shape[0], -1)                  # (N, 4 n_freq)


class ContinuumSpinorPsiformer(fnn.Module):
    """Single generalized-spinor-determinant Psiformer. See module docstring."""
    n_elec: int
    Gs: Any               # (n_G, 2) float    -- envelope plane waves
    W: Any                # (n_orb, n_G, 2) complex -- phi_a(r,s) = sum_G W[a,G,s] exp(i Gs[G].r)
    L: float
    embed_dim: int = 32
    depth: int = 3
    n_heads: int = 4
    ff_mult: int = 2
    n_freq: int = 4
    n_dets: int = 1
    learnable_env: bool = False
    use_jastrow: bool = False
    jastrow_kind: str = "rbf"   # "rbf" = flexible local-basis pair Jastrow (Valenti B-spline class); "cusp"
    n_knots: int = 10           # RBF knots for the flexible Jastrow
    cusp: float = 1.0     # opposite-spin cusp slope u'(0) = m* q_coulomb / (d-1); =1 for m*=1,q=1,d=2

    @fnn.compact
    def __call__(self, R, S):
        n_orb = self.n_elec
        Gs = jnp.asarray(self.Gs, jnp.float64)
        W = jnp.asarray(self.W, jnp.complex128)
        if self.learnable_env:
            dW = self.param("dW", lambda k: jnp.zeros_like(W))
            W = W + dW

        # --- token features: periodic position + spin one-hot ---
        pos = periodic_features(R, self.L, self.n_freq)                 # (N, 4 n_freq)
        spin = jax.nn.one_hot(S, 2, dtype=jnp.float64)                  # (N, 2)
        x = jnp.concatenate([pos, spin], axis=-1)
        x = fnn.Dense(self.embed_dim, param_dtype=jnp.float64, name="embed")(x)

        # --- attention backbone (real) ---
        for _ in range(self.depth):
            x = TransformerBlock(self.embed_dim, self.n_heads, self.ff_mult)(x)
        x = fnn.LayerNorm()(x)                                          # (N, embed_dim)

        # --- complex backflow, ZERO-INIT: bf (n_dets, n_orb, N) ---
        bf = fnn.Dense(self.n_dets * n_orb, param_dtype=jnp.complex128, name="backflow",
                       kernel_init=fnn.initializers.zeros,
                       bias_init=fnn.initializers.zeros)(x.astype(jnp.complex128))
        bf = bf.reshape(self.n_elec, self.n_dets, n_orb)               # (N, n_dets, n_orb)
        bf = jnp.transpose(bf, (1, 2, 0))                              # (n_dets, n_orb, N)

        # --- spinor envelope: env[a,i] = sum_G W[a,G,s_i] exp(i Gs[G].r_i) ---
        phase = jnp.exp(1j * (Gs @ R.T))                              # (n_G, N)
        Ws = W[:, :, S]                                                # (n_orb, n_G, N)
        env = jnp.einsum("gi,agi->ai", phase, Ws)                     # (n_orb, N)

        # --- per-determinant matrices, log-sum-exp over dets ---
        M = env[None] * (1.0 + bf)                                     # (n_dets, n_orb, N)
        signs, logabs = jnp.linalg.slogdet(M)                         # (n_dets,)
        logdets = logabs + jnp.log(signs)
        if self.n_dets == 1:
            logpsi = logdets[0]
        else:
            w = self.param("det_logw", fnn.initializers.zeros, (self.n_dets,), jnp.complex128)
            logpsi = jax.scipy.special.logsumexp(logdets + w, b=1.0)

        # --- optional two-body Jastrow (REAL, SYMMETRIC, minimum-image, spin-dependent). The SOC phase stays
        #     in the determinant. Two forms: ---
        if self.use_jastrow:
            A = jnp.asarray(square_cell(self.L) if np.ndim(self.L) == 0 else np.asarray(self.L, float))
            d = R[:, None, :] - R[None, :, :]                         # (N,N,2)
            dfrac = d @ jnp.linalg.inv(A).T
            dmin = (dfrac - jnp.round(dfrac)) @ A.T                   # nearest-image displacement (round: folx-ok, deriv 0 a.e.)
            r = jnp.sqrt(jnp.sum(dmin * dmin, -1) + 1e-12)           # (N,N)
            same = (S[:, None] == S[None, :])
            if self.jastrow_kind == "rbf":
                # FLEXIBLE local-basis (Gaussian-RBF) pair Jastrow -- the class of Valenti's B-spline Jastrow
                # (arXiv:2512.07947). u(r)=sum_k c_{spin,k} exp(-((r-mu_k)/w)^2); ZERO-INIT c (starts as an
                # exact no-op -> clean SR start), LINEAR in c on a LOCAL basis (well-conditioned/stable,
                # unlike the earlier cosine-MLP); spin-dependent. Captures the full pair correlation hole,
                # not just the r->0 cusp.
                area = float(abs(np.linalg.det(square_cell(self.L) if np.ndim(self.L) == 0 else np.asarray(self.L, float))))
                rcut = 0.5 * np.sqrt(area)
                mu = jnp.asarray(np.linspace(0.0, rcut, self.n_knots))          # (K,)
                wdt = rcut / max(self.n_knots - 1, 1)
                rbf = jnp.exp(-((r[..., None] - mu) / wdt) ** 2)                # (N,N,K)
                c_same = self.param("jas_c_same", fnn.initializers.zeros, (self.n_knots,), jnp.float64)
                c_opp = self.param("jas_c_opp", fnn.initializers.zeros, (self.n_knots,), jnp.float64)
                uij = jnp.where(same, (rbf * c_same).sum(-1), (rbf * c_opp).sum(-1))
            else:  # "cusp": fixed 2D Kato cusp slope u'(0)=self.cusp, learnable range a (bounds E_loc at r->0)
                a = jax.nn.softplus(self.param("jastrow_loga", lambda k: jnp.array(0.0, jnp.float64))) + 0.05
                alpha = jnp.where(same, self.cusp / 3.0, self.cusp)
                uij = alpha * a * (1.0 - jnp.exp(-r / a))
            iu = jnp.triu_indices(self.n_elec, 1)
            logpsi = logpsi + jnp.sum(uij[iu]).astype(logpsi.dtype)
        return logpsi


def make_logpsi(model, params):
    """Bind params -> logpsi(R, S) -> complex, matching the local_energy / sampler interface."""
    def logpsi(R, S):
        return model.apply(params, R, S)
    return logpsi
