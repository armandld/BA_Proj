"""Physics-derived Hamiltonian coefficients for the AMR decision.

The default ``norm="max"`` uses dimensionless, bounded terms:

* ``C``: vector jumps, with ``max(abs(C)) = w_zz``;
* ``K``: vorticity, current and optional X-point signals, each normalized
  independently and then combined with ``max(abs(sum(K))) = w_zzzz``;
* ``H``: the classical score bias, scaled by the largest coupling.

Signals below a relative floating-point round-off bound are treated as zero.
``norm="legacy"`` is retained only to reproduce frozen historical artifacts.
"""

import numpy as np

from Simulation.grid import curl_z


class PhysicalMapperV2:
    """Map MHD fields and a classical score to dimensionless coefficients.

    The mapper depends on relative spatial structure, not on viscosity or
    resistivity. Uniform rescaling of the fields leaves the default ``max``
    coefficients unchanged.
    """

    # Fixed design weights.
    W_ZZ = 2.0       # ZZ coupling weight
    W_ZZZZ = 1.0     # ZZZZ coupling weight
    C_BIAS = 0.1     # Z bias scale relative to the largest coupling.
    EPS = 1e-10       # additive guard used by the legacy formulas only
    ROUND_OFF_FACTOR = 128.0

    # ``legacy`` is read-only compatibility for frozen artifacts.
    NORMALISATIONS = ("legacy", "max")

    def __init__(self, dx=1.0, c_bias=None, w_zz=None, w_zzzz=None,
                 fixed_curl=True, norm="max"):
        """
        Parameters
        ----------
        dx : float
            Grid cell spacing.
        c_bias : float, optional
            Z-bias scale relative to the largest effective coupling. If
            omitted, use the a-priori constant ``C_BIAS=0.1``.
        w_zz, w_zzzz : float, optional
            Override ZZ / ZZZZ coupling weights (rarely needed).
        """
        self.dx = float(dx)
        if not np.isfinite(self.dx) or self.dx <= 0.0:
            raise ValueError(f"dx must be finite and positive, got {dx!r}")
        if not isinstance(fixed_curl, (bool, np.bool_)):
            raise TypeError("fixed_curl must be a boolean")
        self.fixed_curl = bool(fixed_curl)
        if norm not in self.NORMALISATIONS:
            raise ValueError(
                f"norm={norm!r} inconnue ; attendu l'une de "
                f"{self.NORMALISATIONS}")
        self.norm = norm
        self.c_bias = self.C_BIAS if c_bias is None else float(c_bias)
        self.w_zz = self.W_ZZ if w_zz is None else float(w_zz)
        self.w_zzzz = self.W_ZZZZ if w_zzzz is None else float(w_zzzz)
        if not np.isfinite(self.c_bias) or self.c_bias < 0.0:
            raise ValueError("c_bias must be finite and non-negative")
        if not np.isfinite(self.w_zz) or self.w_zz <= 0.0:
            raise ValueError("w_zz must be finite and positive")
        if not np.isfinite(self.w_zzzz) or self.w_zzzz <= 0.0:
            raise ValueError("w_zzzz must be finite and positive")

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
        sim : MHDSolver or None
            Unused compatibility argument shared with ``PhysicalMapper``.
        score : (N, N) array
            Classical instability score in [0, 1].
        fields : dict
            Keys: 'vx', 'vy', 'Bx', 'By', 'Jz'.
        threshold_amr : float
            Refinement threshold (the one free parameter).
        advanced_anomalies_enabled : bool
            Include the X-point reconnection signal when true.
        dx_override : float, optional
            Effective cell size for downsampled fields.

        Returns
        -------
        dict with 'H_edges', 'C_edges', 'K_plaquettes', ['K_xpoint'],
             'threshold_amr', 'w_z_frac'

             Array-valued coefficient keys form the energy-scale contract
             consumed by ``call_vqa_shell.py``.
        """
        dx = self.dx if dx_override is None else float(dx_override)
        if not np.isfinite(dx) or dx <= 0.0:
            raise ValueError(f"dx must be finite and positive, got {dx!r}")
        if not isinstance(advanced_anomalies_enabled, (bool, np.bool_)):
            raise TypeError("advanced_anomalies_enabled must be a boolean")
        if not np.isfinite(threshold_amr):
            raise ValueError("threshold_amr must be finite")

        vx, vy = (np.asarray(fields[key]) for key in ('vx', 'vy'))
        Bx, By = (np.asarray(fields[key]) for key in ('Bx', 'By'))
        score = np.asarray(score)
        shapes = {array.shape for array in (vx, vy, Bx, By, score)}
        if len(shapes) != 1 or any(array.ndim != 2
                                   for array in (vx, vy, Bx, By, score)):
            raise ValueError(
                "vx, vy, Bx, By and score must be two-dimensional arrays "
                "with the same shape")
        if not all(np.all(np.isfinite(array))
                   for array in (vx, vy, Bx, By, score)):
            raise ValueError("fields and score must contain only finite values")

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

        if self.norm == "max":
            pic = max(float(np.max(jump_h)), float(np.max(jump_v)))
            jump_floor = self._difference_roundoff_floor(vx, vy, Bx, By)
            norm_jump = pic if pic > jump_floor else 1.0
            if pic <= jump_floor:
                jump_h = np.zeros_like(jump_h)
                jump_v = np.zeros_like(jump_v)
        else:
            norm_jump = 0.5 * (np.mean(jump_h) + np.mean(jump_v)) + self.EPS

        C_horiz = -self.w_zz * jump_h / norm_jump
        C_vert = -self.w_zz * jump_v / norm_jump

        # ==============================================================
        #  2. ZZZZ (circulation plaquette)
        #     K_p = -w_ZZZZ * (|omega_z,p| + |J_z,p|) / (max|omega| + max|J| + eps)
        # ==============================================================

        # discrete vorticity: omega_z = dvy/dx - dvx/dy
        omega_z = curl_z(vx, vy, self.fixed_curl)

        # discrete current density: J_z = dBy/dx - dBx/dy
        Jz_curl = curl_z(Bx, By, self.fixed_curl)

        # All active plaquette signals share one final family normalization.
        xpoint_signal = None
        det_floor = 0.0
        if advanced_anomalies_enabled:
            jacobian = self._jacobian_B(Bx, By, dx)
            det_J_B = jacobian[0] * jacobian[3] - jacobian[1] * jacobian[2]
            det_floor = self._determinant_roundoff_floor(
                Bx, By, dx, jacobian)
            xpoint_signal = np.maximum(0.0, -det_J_B)

        def _adim(signal, noise_floor):
            """Normalize a resolved signal and discard round-off residue."""
            pic = float(np.max(np.abs(signal)))
            if pic <= noise_floor:
                return np.zeros_like(signal, dtype=float)
            return np.abs(signal) / pic

        if self.norm == "max":
            # Vorticity, current and X-point structure share one plaquette
            # family. Each signal is made dimensionless independently, then
            # their sum is bounded by ``w_zzzz``. ``K_xpoint`` remains a
            # separate key so term ablations can address it explicitly.
            omega_floor = self._difference_roundoff_floor(vx, vy)
            current_floor = self._difference_roundoff_floor(Bx, By)
            signal_plaq = (_adim(omega_z, omega_floor)
                           + _adim(Jz_curl, current_floor))
            signal_total = signal_plaq
            if xpoint_signal is not None:
                signal_total = signal_total + _adim(xpoint_signal, det_floor)
            pic_total = float(np.max(signal_total))
            norm_plaq = pic_total if pic_total > 0.0 else 1.0
        else:
            signal_plaq = np.abs(omega_z) + np.abs(Jz_curl)
            norm_plaq = np.max(np.abs(omega_z)) + np.max(np.abs(Jz_curl)) + self.EPS

        K_plaquettes = -self.w_zzzz * signal_plaq / norm_plaq

        K_xpoint = None
        if xpoint_signal is not None:
            if self.norm == "max":
                K_xpoint = (-self.w_zzzz
                            * _adim(xpoint_signal, det_floor) / norm_plaq)
            else:
                max_det = np.max(np.abs(det_J_B)) + self.EPS
                K_xpoint = -self.w_zzzz * xpoint_signal / max_det

        # ==============================================================
        #  3. Z (activity bias)
        #     h_i = +c * max(|C|, |K_effective|) * (score_i - threshold)
        # ==============================================================
        effective_K = (K_plaquettes if K_xpoint is None
                       else K_plaquettes + K_xpoint)
        all_coeffs = np.concatenate([
            np.abs(C_horiz).ravel(),
            np.abs(C_vert).ravel(),
            np.abs(effective_K).ravel(),
        ])
        if self.norm == "max":
            echelle = float(np.max(all_coeffs))
        else:
            nonzero = all_coeffs[all_coeffs > self.EPS]
            echelle = (float(np.median(nonzero)) if len(nonzero)
                       else 0.0)

        z_bias = self.c_bias * echelle * (score - threshold_amr)
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

        if K_xpoint is not None:
            result["K_xpoint"] = K_xpoint

        return result

    # ------------------------------------------------------------------
    #  Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _floating_epsilon(*arrays):
        dtype = np.result_type(*(np.asarray(a).dtype for a in arrays))
        if not np.issubdtype(dtype, np.inexact):
            dtype = np.dtype(float)
        return float(np.finfo(dtype).eps)

    @classmethod
    def _difference_roundoff_floor(cls, *fields):
        """Absolute floor for differences of fields with large offsets."""
        scale = max(float(np.max(np.abs(field))) for field in fields)
        if scale == 0.0:
            return 0.0
        return cls.ROUND_OFF_FACTOR * cls._floating_epsilon(*fields) * scale

    @staticmethod
    def _jacobian_B(Bx, By, dx):
        if not np.isfinite(dx) or dx <= 0.0:
            raise ValueError(f"dx must be finite and positive, got {dx!r}")
        dBx_dx = 0.5 * (np.roll(Bx, -1, axis=0) - np.roll(Bx, 1, axis=0)) / dx
        dBx_dy = 0.5 * (np.roll(Bx, -1, axis=1) - np.roll(Bx, 1, axis=1)) / dx
        dBy_dx = 0.5 * (np.roll(By, -1, axis=0) - np.roll(By, 1, axis=0)) / dx
        dBy_dy = 0.5 * (np.roll(By, -1, axis=1) - np.roll(By, 1, axis=1)) / dx
        return dBx_dx, dBx_dy, dBy_dx, dBy_dy

    @classmethod
    def _determinant_roundoff_floor(cls, Bx, By, dx, jacobian):
        """Round-off bound for ``det(nabla B)`` including cancellation."""
        eps = cls._floating_epsilon(Bx, By)
        field_scale = max(float(np.max(np.abs(Bx))),
                          float(np.max(np.abs(By))))
        derivative_error = 2.0 * eps * field_scale / dx
        a, b, c, d = jacobian
        product_scale = float(np.max(np.abs(a * d) + np.abs(b * c)))
        gradient_scale = max(float(np.max(np.abs(g))) for g in jacobian)
        estimate = (eps * product_scale
                    + 4.0 * derivative_error * gradient_scale
                    + 2.0 * derivative_error ** 2)
        return cls.ROUND_OFF_FACTOR * estimate

    @staticmethod
    def _compute_det_jacobian_B(Bx, By, dx):
        """
        Determinant of the magnetic Jacobian: det(nabla B).

        det(J_B) = dBx/dx * dBy/dy - dBx/dy * dBy/dx

        det < 0 -> X-point (hyperbolic null, reconnection site)
        det > 0 -> O-point (elliptic null, island center)
        """
        a, b, c, d = PhysicalMapperV2._jacobian_B(Bx, By, dx)
        return a * d - b * c


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
