import numpy as np

from Simulation.grid import curl_z, divergence


def _lohner_estimator(f):
    """
    Löhner (1987) second-derivative error estimator.

    E(i,j) = sqrt(f''_xx² + f''_yy²) / (|f'_x| + |f'_y| + eps*|f|)

    Returns a dimensionless field in ~[0, 1] that peaks at discontinuities
    and is insensitive to smooth gradients.

    Les noms `_x` / `_y` ci-dessous suivent la convention inverse de celle du
    depot (axis=1 y est lu comme x). C'est sans consequence numerique : la
    formule est SYMETRIQUE par echange des deux axes, donc echanger les
    etiquettes ne change pas la valeur rendue. `tests/test_mapper_contracts.py`
    fige cette symetrie — c'est elle qui rend le mauvais nom inoffensif, et
    toute edition qui la briserait rendrait le nom dangereux.
    """
    fp_x = np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)
    fp_y = np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)

    fpp_xx = np.roll(f, -1, axis=1) - 2.0 * f + np.roll(f, 1, axis=1)
    fpp_yy = np.roll(f, -1, axis=0) - 2.0 * f + np.roll(f, 1, axis=0)

    numer = np.sqrt(fpp_xx**2 + fpp_yy**2)
    eps_loh = 0.01
    denom = np.abs(fp_x) + np.abs(fp_y) + eps_loh * np.abs(f) + 1e-30
    return numer / denom


class AngleMapper:
    """
    Converts MHD physics into quantum circuit angles.

    Architecture (v4 — Classical Score Initialization):
    ====================================================
    θ (theta) is derived from a multi-indicator classical score:
        score = RMS(vorticity, divergence, |Jz|, Löhner)
        θ = 2 * arcsin(√score)   →   P(|1⟩) = sin²(θ/2) = score

    This gives each qubit the SAME initial probability of being flagged
    as the classical AMR detector.  The QAOA then refines this starting
    point using entanglement (plaquette/vertex topology).

    ψ (psi) encodes the temporal evolution of the stress flux:
        ψ = (π/2) * tanh(β * ΔΦ / ⟨|ΔΦ|⟩)
    providing the QAOA with rate-of-change information.

    The stress flux Φ is still computed (needed for:
      - Hamiltonian H_edges coefficient: H = 4*(threshold - cos(2*arctan(α*Φ/⟨Φ⟩)))
      - ψ phase encoding: temporal derivative of Φ
    ) but is NO LONGER used for θ.
    """

    def __init__(self, v0=1.0, B0=1.0, J0=1, w_compress=2.0, w_shear=1.0,
                 w_J=1.0, fixed_flux=True):
        self.v0 = v0
        self.B0 = B0
        self.J0 = J0
        self.w_compress = w_compress
        self.w_shear = w_shear
        self.w_J = w_J
        # Voir compute_stress_flux : False reproduit bit-a-bit le chemin
        # historique (normale/tangentielle echangees), True applique la
        # convention AXIS_X/AXIS_Y declaree dans Simulation.grid.
        self.fixed_flux = bool(fixed_flux)

    # ── Stress flux (still needed for Hamiltonian + ψ) ─────────────────

    def _compute_filtered_flux(self, array):
        """
        Shock-Diode logic on field differences (d_norm_v, d_tang_v, d_norm_B, d_tang_B, d_Jz).
        Compression (negative normal) is dangerous; expansion is ignored.
        Shear (tangential) is always dangerous.
        """
        shock_v = np.maximum(0, -array[0])
        shock_B = np.maximum(0, -array[2])
        total_shock = np.sqrt(shock_v**2 + shock_B**2)

        shear_v = np.abs(array[1])
        shear_B = np.abs(array[3])
        total_shear = np.sqrt(shear_v**2 + shear_B**2)

        return np.sqrt(
            (self.w_compress * total_shock)**2
            + (self.w_shear * total_shear)**2
            + (self.w_J * array[4])**2
        )

    def compute_stress_flux(self, physics_state):
        """
        Computes edge-wise MHD stress flux Φ.

        Une famille d'aretes est nommee par l'axe du DECALAGE qui relie les
        deux voisins : 'horizontal' = decalage le long de axis=1, 'vertical'
        = decalage le long de axis=0. Sous la convention du depot
        (`grid.AXIS_X = 0`, `AXIS_Y = 1`) la normale d'une arete 'horizontale'
        est donc y, et celle d'une arete 'verticale' est x.

        `_compute_filtered_flux` lit array[0] comme la composante NORMALE
        (diode de choc, poids w_compress) et array[1] comme la TANGENTIELLE
        (poids w_shear). L'ordre des tuples ci-dessous doit donc suivre l'axe
        du decalage.

        fixed_flux=False conserve l'ordre historique, ecrit sous la convention
        inverse (axis=1 lu comme x). Il place la composante transverse dans la
        case de la normale : la diode de choc s'applique alors au cisaillement
        — dont le signe ne porte aucune information de compression — et la
        compression passe par `np.abs` dans la branche de cisaillement. Sur des
        champs analytiques ce chemin rend un rapport compression/cisaillement
        de 0.5 la ou la conception en demande 2.0, et rend la diode inerte
        (compression et expansion donnent le meme flux).

        Returns dict with:
            'phi_horizontal', 'phi_vertical'            — flux magnitudes
        """
        vx = physics_state['vx'] / self.v0
        vy = physics_state['vy'] / self.v0
        Bx = physics_state['Bx'] / self.B0
        By = physics_state['By'] / self.B0
        Jz = physics_state['Jz'] / self.J0

        if self.fixed_flux:
            # axis=1 = AXIS_Y : normale = y  -> (vy, vx, By, Bx)
            order_h = (vy, vx, By, Bx, Jz)
            # axis=0 = AXIS_X : normale = x  -> (vx, vy, Bx, By)
            order_v = (vx, vy, Bx, By, Jz)
        else:
            order_h = (vx, vy, Bx, By, Jz)
            order_v = (vy, vx, By, Bx, Jz)

        # Aretes 'horizontales' : voisins decales le long de axis=1
        def _h_diffs(arr, shift):
            return np.roll(arr, shift, axis=1) - arr

        grad_h_right = np.array([_h_diffs(f, -1) for f in order_h])
        grad_h_left = np.array([_h_diffs(f, 1) for f in order_h])
        norm_flux_h = np.maximum(
            self._compute_filtered_flux(grad_h_right),
            self._compute_filtered_flux(grad_h_left),
        )

        # Aretes 'verticales' : voisins decales le long de axis=0
        def _v_diffs(arr, shift):
            return np.roll(arr, shift, axis=0) - arr

        grad_v_bottom = np.array([_v_diffs(f, -1) for f in order_v])
        grad_v_top = np.array([_v_diffs(f, 1) for f in order_v])
        norm_flux_v = np.maximum(
            self._compute_filtered_flux(grad_v_bottom),
            self._compute_filtered_flux(grad_v_top),
        )

        return {
            'phi_horizontal': norm_flux_h,
            'phi_vertical': norm_flux_v,
        }

    # ── Classical multi-indicator score → θ ────────────────────────────

    @staticmethod
    def classical_score(physics_state, fixed_curl=True):
        """
        Compute the multi-indicator instability score on the full domain.

        Combines 4 standard MHD-AMR criteria (same as FLASH / Athena++ / PLUTO):
          1. |ωz|  = |∂vy/∂x − ∂vx/∂y|   (vorticity)
          2. |∇·v| = |∂vx/∂x + ∂vy/∂y|   (compression)
          3. |Jz|                           (current density)
          4. Löhner estimator on |B|        (discontinuity sensor)

        Each indicator is normalized to [0, 1] by its domain-wide max.
        The score is the RMS of the 4 normalized indicators:
            score = √((s_vort² + s_div² + s_jz² + s_loh²) / 4)

        Returns (N, N) array in [0, 1].
        """
        vx, vy = physics_state['vx'], physics_state['vy']
        Bx, By = physics_state['Bx'], physics_state['By']
        Jz = physics_state['Jz']

        # Indicator 1: Vorticity |ωz|
        vorticity = np.abs(curl_z(vx, vy, fixed_curl))

        # Indicator 2: Velocity divergence |∇·v|
        div_v = np.abs(divergence(vx, vy, fixed_curl))

        # Indicator 3: Current density |Jz|
        abs_Jz = np.abs(Jz)

        # Indicator 4: Löhner estimator on |B|
        B_mag = np.sqrt(Bx**2 + By**2)
        lohner_B = _lohner_estimator(B_mag)

        # Normalize each to [0, 1]
        def _norm(arr):
            mx = np.max(arr)
            return arr / mx if mx > 1e-12 else arr

        s_vort = _norm(vorticity)
        s_div = _norm(div_v)
        s_jz = _norm(abs_Jz)
        s_loh = _norm(lohner_B)

        score = np.sqrt((s_vort**2 + s_div**2 + s_jz**2 + s_loh**2) / 4.0)
        return np.clip(score, 0.0, 1.0)

    # ── Angle encoding ─────────────────────────────────────────────────

    def map_to_angles(self, score_h, score_v, phi_dict_prev, phi_dict,
                      AveragePhiDev, beta):
        """
        Converts classical score + stress flux into qubit rotation angles.

        Parameters
        ----------
        score_h, score_v : (N, N) arrays
            Classical multi-indicator score in [0, 1], one per edge direction.
            Used for θ: P(|1⟩) = sin²(θ/2) = score.

            La signature accepte deux cartes distinctes, mais TOUS les
            appelants du depot passent la meme (`refinement._prepare_vqa_input`
            passe `mini_score` deux fois). En deploiement θ_h ≡ θ_v : les deux
            familles de qubits partent du meme etat, et ne se distinguent que
            par ψ et par C_horiz / C_vert dans l'Hamiltonien.
            `tests/test_mapper_contracts.py` fige cette equivalence : si un
            appelant venait a passer deux cartes differentes, ce serait un
            changement de comportement scientifique, pas un detail.
        phi_dict_prev, phi_dict : dict or None
            Previous and current stress flux dicts (for ψ temporal encoding).
        AveragePhiDev : float or None
            Global mean |ΔΦ| for ψ normalization.
        beta : float
            Phase encoding sensitivity.

        Returns
        -------
        (theta_h, theta_v, psi_h, psi_v) : tuple of (N, N) arrays
        """
        # θ = 2·arcsin(√score)  →  sin²(θ/2) = score  →  P(|1⟩) = score
        theta_h = 2.0 * np.arcsin(np.sqrt(np.clip(score_h, 0.0, 1.0)))
        theta_v = 2.0 * np.arcsin(np.sqrt(np.clip(score_v, 0.0, 1.0)))

        # ψ from temporal evolution of stress flux (unchanged)
        if phi_dict_prev is None or phi_dict is None:
            return theta_h, theta_v, np.zeros_like(theta_h), np.zeros_like(theta_v)

        psi_h = self._activation_function_psi(
            phi_dict_prev['phi_horizontal'], phi_dict['phi_horizontal'],
            beta, AveragePhiDev,
        )
        psi_v = self._activation_function_psi(
            phi_dict_prev['phi_vertical'], phi_dict['phi_vertical'],
            beta, AveragePhiDev,
        )
        return theta_h, theta_v, psi_h, psi_v

    def _activation_function_psi(self, phi_prev, phi, beta, AveragePhiDev):
        """
        Phase encoding: ψ = (π/2) · tanh(β · ΔΦ / ⟨|ΔΦ|⟩)
        """
        if phi_prev is None or AveragePhiDev is None or AveragePhiDev < 1e-12:
            return np.zeros_like(phi)
        return (np.pi / 2) * np.tanh(beta * (phi - phi_prev) / AveragePhiDev)
