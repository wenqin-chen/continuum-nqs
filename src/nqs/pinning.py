"""Pinning fields that NUCLEATE each competing order in the NQS (reuse the local-energy ext_field).

A pinning field adds  H_pin = -h_pin sum_i n_target(r_i) . sigma_i  to the Hamiltonian (i.e.
ext_field(r) = h_pin * n_target(r), a unit-vector target). Early training with h_pin>0 imposes the
target order; annealing h_pin -> 0 then tests whether it SURVIVES on its own (the decisive check).

Targets (note: Rashba SOC favors NEEL/cycloidal textures, not Bloch):
  skx        -- triple-Q NEEL skyrmion (in-plane RADIAL along q_hat_j), topological (Berg-Luscher != 0);
                skx_bloch = the Bloch (tangential z_hat x q_hat) control, Rashba-DISfavored.
  spiral     -- single-Q CYCLOIDAL spiral (spin rotates in the z_hat-q_hat plane); spiral_xy = flat in-plane
                XY-helix control (the old default, Rashba-neutral).
  overhauser -- collinear z-SDW (Overhauser competitor).
  fm         -- uniform axis (default zhat).
  fluid      -- no pinning (ext_field = None; the uniform Rashba liquid).
"""
import numpy as np
import jax.numpy as jnp


def make_skx_pin(q1, q2, h_pin, m0=0.3, chirality=1.0, neel=True):
    """ext_field(r) = h_pin * unit triple-Q skyrmion texture (arms q1,q2,q3=-(q1+q2)).
    neel=True (DEFAULT since 2026-07-06): in-plane spin RADIAL (along q_hat) -- the Rashba-FAVORED
    Neel/cycloidal skyrmion. neel=False: in-plane TANGENTIAL (z_hat x q_hat = (-qy,qx)) -- Bloch, which
    Rashba DISfavors (control only). """
    qs = jnp.asarray(np.stack([q1, q2, -(np.asarray(q1) + np.asarray(q2))]))   # (3,2)
    qhat = qs / jnp.linalg.norm(qs, axis=1, keepdims=True)
    ehat = qhat if neel else jnp.stack([-qhat[:, 1], qhat[:, 0]], axis=1)      # radial (Neel) vs tangential (Bloch)

    def ext_field(r):
        qr = qs @ r                                       # (3,)
        s = jnp.sin(qr)
        mz = m0 + jnp.sum(jnp.cos(qr))
        mx = chirality * jnp.sum(s * ehat[:, 0])
        my = chirality * jnp.sum(s * ehat[:, 1])
        m = jnp.array([mx, my, mz])
        return h_pin * m / (jnp.linalg.norm(m) + 1e-12)
    return ext_field


def make_fm_pin(h_pin, axis=(0.0, 0.0, 1.0)):
    ax = jnp.asarray(axis, float)
    ax = ax / jnp.linalg.norm(ax)

    def ext_field(r):
        return h_pin * ax
    return ext_field


def make_spiral_pin(Q, h_pin, cycloidal=True):
    """Single-Q spiral. cycloidal=True (DEFAULT since 2026-07-06): spin rotates in the z_hat-q_hat plane --
    the Rashba-FAVORED cycloid m(r) = (sin(Q.r) q_hat, cos(Q.r)). cycloidal=False: the old flat in-plane
    XY-helix m=(cos,sin,0), Rashba-neutral (control). """
    Q = jnp.asarray(Q, float)
    qh = Q / (jnp.linalg.norm(Q) + 1e-12)

    def ext_field(r):
        ph = Q @ r
        if cycloidal:
            return h_pin * jnp.array([jnp.sin(ph) * qh[0], jnp.sin(ph) * qh[1], jnp.cos(ph)])
        return h_pin * jnp.array([jnp.cos(ph), jnp.sin(ph), 0.0])
    return ext_field


def make_overhauser_pin(Q, h_pin):
    """Collinear z-SDW -- the Overhauser competitor (2k_F SDW+CDW; here the spin part):
    ext_field(r) = h_pin * (0, 0, cos(Q.r))."""
    Q = jnp.asarray(Q, float)

    def ext_field(r):
        return h_pin * jnp.array([0.0, 0.0, jnp.cos(Q @ r)])
    return ext_field


def load_hf_texture(path):
    """Load a gen_cellD_texture .npz -> the HF spin texture + geometry for the faithful SkX seed.
    Also returns the CHARGE field (n_keys/n_vals) if present (add_charge_field.py) -- the Hartree-augmented
    warm-start needed at higher filling where the spin-only texture fills a topologically trivial manifold."""
    d = np.load(path)
    out = dict(q1=d["q1"], q2=d["q2"], s_keys=d["s_keys"], s_vals=d["s_vals"],
               BL=float(d["BL"]), E_var=float(d["E_var"]), q_star=float(d["q_star"]),
               e_per_cell=float(d["e_per_cell"]))
    if "n_keys" in d.files:
        out["n_keys"] = d["n_keys"]; out["n_vals"] = d["n_vals"]
    return out


def make_hf_skx_pin(s_keys, s_vals, q1, q2, h_pin):
    """Pinning field from a CONVERGED HF spin texture -- a converged Hartree-Fock texture:
    n(r) = Re sum_G s_field[G] exp(i G.r),  G = m q1 + n q2;  ext_field = h_pin * n(r)/|n(r)|.
    s_keys (nG,2) int (m,n) harmonics; s_vals (nG,3) complex spin Fourier components."""
    q1 = np.asarray(q1, float)
    q2 = np.asarray(q2, float)
    Gs = jnp.asarray(np.array([m * q1 + n * q2 for (m, n) in np.asarray(s_keys)]))   # (nG,2)
    S = jnp.asarray(np.asarray(s_vals, dtype=complex))                                # (nG,3)

    def ext_field(r):
        phase = jnp.exp(1j * (Gs @ r))                       # (nG,)
        m = jnp.real(jnp.sum(phase[:, None] * S, axis=0))    # (3,) real-space texture
        return h_pin * m / (jnp.linalg.norm(m) + 1e-12)
    return ext_field


def make_pin(kind, q1, q2, h_pin, **kw):
    """Dispatch by competitor name. q1,q2 = ordering arms (used by skx/spiral/overhauser)."""
    if kind == "fluid":
        return None                            # uniform Rashba liquid, no pin
    if kind == "fm":
        return make_fm_pin(h_pin, **kw)
    if kind == "spiral":
        return make_spiral_pin(q1, h_pin)                       # cycloidal (Rashba-favored)
    if kind == "spiral_xy":
        return make_spiral_pin(q1, h_pin, cycloidal=False)      # flat in-plane helix (control)
    if kind == "overhauser":
        return make_overhauser_pin(q1, h_pin)
    if kind == "skx":
        return make_skx_pin(q1, q2, h_pin, **kw)                # Neel (Rashba-favored)
    if kind == "skx_bloch":
        return make_skx_pin(q1, q2, h_pin, neel=False, **kw)    # Bloch (control)
    raise ValueError(kind)


def berg_luscher_realspace(field_fn, A, ngrid=24):
    """Lattice Berg-Luscher charge of a real-space unit-field n(r)=field/|field| over the cell A
    (columns = lattice vectors). For checking that the skx target is topological."""
    A = np.asarray(A, float)
    fr = (np.arange(ngrid) + 0.5) / ngrid
    n = np.zeros((ngrid, ngrid, 3))
    for i, u in enumerate(fr):
        for j, v in enumerate(fr):
            r = A @ np.array([u, v])
            m = np.array(field_fn(jnp.asarray(r)))
            n[i, j] = m / (np.linalg.norm(m) + 1e-12)
    tot = 0.0
    for i in range(ngrid):
        for j in range(ngrid):
            a = n[i, j]; b = n[(i + 1) % ngrid, j]
            c = n[i, (j + 1) % ngrid]; d = n[(i + 1) % ngrid, (j + 1) % ngrid]
            for (p, q, w) in [(a, b, d), (a, d, c)]:
                num = np.dot(p, np.cross(q, w))
                den = 1 + np.dot(p, q) + np.dot(q, w) + np.dot(w, p)
                tot += 2 * np.arctan2(num, den)
    return tot / (4 * np.pi)
