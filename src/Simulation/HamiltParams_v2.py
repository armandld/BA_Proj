"""
Physics-first Hamiltonian coefficients (v2) -- parameter-free.

All coefficients are derived directly from the physical fields with
simple domain-normalized ratios. No trainable parameters live inside
the Hamiltonian; only thr_amr (the refinement threshold) remains as
a physical choice.

Architecture:
=============
  ZZ (gradient coupling):
    C_ij = -w_ZZ * sqrt(|dv_ij|^2 + |dB_ij|^2) / (<sqrt(|dv|^2 + |dB|^2)> + eps)

  ZZZZ (circulation plaquette):
    K_p = -w_ZZZZ * (|omega_z,p| + |J_z,p|) / (|omega_z|_max + |J_z|_max + eps)

  ZZZZ X-point (optional):
    K_xp = -w_ZZZZ * max(0, -det(nabla B)) / (|det(nabla B)|_max + eps)

  Z (bias):
    h_i = -c * median(|C|, |K|) * (s_i - thr)
    where c = 0.1 fixed

Fixed weights: w_ZZ = 2, w_ZZZZ = 1. Negative signs = ferromagnetic.

Phase encoding (for AngleMapper_v2):
  phi_ij = (pi/2) * tanh(delta_Phi_ij / (<|delta_Phi|> + eps))
  No beta parameter -- tanh normalises naturally.

Differences from v1 (HamiltParams.py):
---------------------------------------
  - Removed: gamma_hydro, gamma_mag (f-gate growth rates)
  - Removed: kappa (leaky sigmoid steepness)
  - Removed: sigma (Gaussian uncertainty width)
  - Removed: beta_curl, beta_xpoint (Michelson sensitivity)
  - Removed: w_z_frac as tuneable (fixed at c=0.1)
  - Removed: f-gate, g-gate, threshold-contrast, Gaussian weighting
  - Added: simple domain-normalized ratios (mean for ZZ, max for ZZZZ)
  - Result: 0 trainable parameters in Hamiltonian (was ~8)
"""

import numpy as np


class PhysicalMapperV2:
    """
    Parameter-free Hamiltonian coefficient computation from MHD fields.

    Only physical constants (nu, eta, dx) and the refinement threshold
    (thr_amr) affect the output. No trainable hyperparameters.
    """

    # Fixed weights (not trained, chosen once by physical reasoning)
    W_ZZ = 2.0       # ZZ coupling weight
    W_ZZZZ = 1.0     # ZZZZ coupling weight
    C_BIAS = 0.1     # Z bias scale: fraction of median(|C|,|K|). Default.
    EPS = 1e-10       # division-by-zero guard

    def __init__(self, dx=1.0, c_bias=None, w_zz=None, w_zzzz=None):
        """
        Parameters
        ----------
        dx : float
            Grid cell spacing.
        c_bias : float, optional
            Z-bias scale (fraction of median(|C|,|K|)). If None, use the
            class default C_BIAS=0.1. Raising this strengthens the local
            bias relative to the ferromagnetic couplings — phase 9 sweeps
            this to find the value where the Hamiltonian's ground state
            actually matches the hard-patch mask.
        w_zz, w_zzzz : float, optional
            Override ZZ / ZZZZ coupling weights (rarely needed).
        """
        self.dx = dx
        self.c_bias = self.C_BIAS if c_bias is None else float(c_bias)
        self.w_zz = self.W_ZZ if w_zz is None else float(w_zz)
        self.w_zzzz = self.W_ZZZZ if w_zzzz is None else float(w_zzzz)

    # ------------------------------------------------------------------
    #  Main computation
    # ------------------------------------------------------------------

    def compute_coefficients(self, sim, score, fields, threshold_amr,
                             advanced_anomalies_enabled=False,
                             dx_override=None, **kwargs):
        """
        Compute Hamiltonian coefficients (Z + ZZ + ZZZZ).

        Parameters
        ----------
        sim : MHDSolver
            Solver instance (grid methods used for gradient operators).
        score : (N, N) array
            Classical instability score in [0, 1].
        fields : dict
            Keys: 'vx', 'vy', 'Bx', 'By', 'Jz'.
        threshold_amr : float
            Refinement threshold (the one free parameter).
        advanced_anomalies_enabled : bool
            Whether to compute X-point reconnection terms.
        dx_override : float, optional
            Effective cell size for downsampled fields.

        Returns
        -------
        dict with 'H_edges', 'C_edges', 'K_plaquettes', ['K_xpoint'],
             'threshold_amr', 'w_z_frac'
        """
        dx = dx_override if dx_override is not None else self.dx

        vx, vy = fields['vx'], fields['vy']
        Bx, By = fields['Bx'], fields['By']

        # ==============================================================
        #  1. ZZ (gradient coupling)
        #     C_ij = -w_ZZ * |jump_ij| / (<|jump|> + eps)
        # ==============================================================

        # vector jump magnitude across horizontal edges (axis=1)
        dvx_h = vx - np.roll(vx, -1, axis=1)
        dvy_h = vy - np.roll(vy, -1, axis=1)
        dBx_h = Bx - np.roll(Bx, -1, axis=1)
        dBy_h = By - np.roll(By, -1, axis=1)
        jump_h = np.sqrt(dvx_h**2 + dvy_h**2 + dBx_h**2 + dBy_h**2)

        # vector jump magnitude across vertical edges (axis=0)
        dvx_v = vx - np.roll(vx, -1, axis=0)
        dvy_v = vy - np.roll(vy, -1, axis=0)
        dBx_v = Bx - np.roll(Bx, -1, axis=0)
        dBy_v = By - np.roll(By, -1, axis=0)
        jump_v = np.sqrt(dvx_v**2 + dvy_v**2 + dBx_v**2 + dBy_v**2)

        # domain-average normalisation
        mean_jump = 0.5 * (np.mean(jump_h) + np.mean(jump_v)) + self.EPS

        C_horiz = -self.w_zz * jump_h / mean_jump
        C_vert = -self.w_zz * jump_v / mean_jump

        # ==============================================================
        #  2. ZZZZ (circulation plaquette)
        #     K_p = -w_ZZZZ * (|omega_z,p| + |J_z,p|) / (max|omega| + max|J| + eps)
        # ==============================================================

        # discrete vorticity: omega_z = dvy/dx - dvx/dy
        omega_z = (
            (np.roll(vy, -1, axis=1) - vy)
            - (np.roll(vx, -1, axis=0) - vx)
        )

        # discrete current density: J_z = dBy/dx - dBx/dy
        Jz_curl = (
            (np.roll(By, -1, axis=1) - By)
            - (np.roll(Bx, -1, axis=0) - Bx)
        )

        # domain-max normalisation
        max_omega = np.max(np.abs(omega_z))
        max_Jz = np.max(np.abs(Jz_curl))
        norm_plaq = max_omega + max_Jz + self.EPS

        K_plaquettes = -self.w_zzzz * (np.abs(omega_z) + np.abs(Jz_curl)) / norm_plaq

        # ==============================================================
        #  3. Z (activity bias)
        #     h_i = -c * median(|C|, |K|) * (s_i - thr)
        # ==============================================================
        all_coeffs = np.concatenate([
            np.abs(C_horiz).ravel(),
            np.abs(C_vert).ravel(),
            np.abs(K_plaquettes).ravel(),
        ])
        nonzero = all_coeffs[all_coeffs > self.EPS]
        median_scale = float(np.median(nonzero)) if len(nonzero) > 0 else 0.0

        z_bias = self.c_bias * median_scale * (score - threshold_amr)
        H_horiz = z_bias.copy()
        H_vert = z_bias.copy()

        # ==============================================================
        #  4. X-point reconnection (optional ZZZZ)
        #     K_xp = -w_ZZZZ * max(0, -det(nabla B)) / (max|det| + eps)
        # ==============================================================
        result = {
            "H_edges": (H_horiz, H_vert),
            "C_edges": (C_horiz, C_vert),
            "K_plaquettes": K_plaquettes,
            "threshold_amr": threshold_amr,
            "w_z_frac": self.c_bias,
        }

        if advanced_anomalies_enabled:
            det_J_B = self._compute_det_jacobian_B(Bx, By, dx)
            xpoint_signal = np.maximum(0.0, -det_J_B)
            max_det = np.max(np.abs(det_J_B)) + self.EPS
            K_xpoint = -self.w_zzzz * xpoint_signal / max_det
            result["K_xpoint"] = K_xpoint

        return result

    # ------------------------------------------------------------------
    #  Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_det_jacobian_B(Bx, By, dx):
        """
        Determinant of the magnetic Jacobian: det(nabla B).

        det(J_B) = dBx/dx * dBy/dy - dBx/dy * dBy/dx

        det < 0 -> X-point (hyperbolic null, reconnection site)
        det > 0 -> O-point (elliptic null, island center)
        """
        dBx_dx = 0.5 * (np.roll(Bx, -1, axis=0) - np.roll(Bx, 1, axis=0)) / dx
        dBx_dy = 0.5 * (np.roll(Bx, -1, axis=1) - np.roll(Bx, 1, axis=1)) / dx
        dBy_dx = 0.5 * (np.roll(By, -1, axis=0) - np.roll(By, 1, axis=0)) / dx
        dBy_dy = 0.5 * (np.roll(By, -1, axis=1) - np.roll(By, 1, axis=1)) / dx
        return dBx_dx * dBy_dy - dBx_dy * dBy_dx


# ======================================================================
#  Phase encoding (v2): parameter-free tanh normalisation
# ======================================================================

def compute_psi_v2(phi_prev, phi, eps=1e-10):
    """
    Parameter-free phase encoding.

    psi = (pi/2) * tanh(delta_Phi / (<|delta_Phi|> + eps))

    No beta parameter -- the tanh normalises naturally by the
    domain-average flux change.

    Parameters
    ----------
    phi_prev, phi : (N, N) arrays
        Previous and current stress flux magnitudes.

    Returns
    -------
    psi : (N, N) array in [-pi/2, pi/2]
    """
    if phi_prev is None or phi is None:
        return np.zeros_like(phi if phi is not None else phi_prev)

    delta = phi - phi_prev
    avg_delta = np.mean(np.abs(delta)) + eps
    return (np.pi / 2.0) * np.tanh(delta / avg_delta)
