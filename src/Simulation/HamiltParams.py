import numpy as np
from Simulation.PhysToAngle import _lohner_estimator
from Simulation.grid import curl_z, divergence, forward_curl_z

class PhysicalMapper:
    """
    Computes spatially-varying Hamiltonian coefficients from MHD fields.

    Architecture (v8 — Decoupled f × g × Mic, corrected signs):
    =============================================================
    Each coefficient is a fully multiplicative product:

      Coeff = ±Weight × g(topology) × f(scale) × Mic(signal)

    **All sign conventions live HERE** — cost_hamiltonian.py uses
    coefficients as-is without any negation.

    Q-criterion convention (from grid._compute_q_criterion):
      Q = 0.5*(omega² - strain²)
      Q > 0 → rotation-dominated (vortices)
      Q < 0 → strain-dominated (shear layers)

    Hamiltonian terms (output signs included):
    -------------------------------------------
    1. **Z (Activity bias)**: REMOVED (v9).
       θ initialization already encodes P(|1⟩) = classical_score.
       With limited QAOA layers (p=1-2), the mixer makes small rotations
       around θ. Adding Z ∝ score would make the QAOA degenerate to a
       classical threshold. By removing Z, the QAOA can ONLY improve on
       the classical init via spatial correlations (ZZ, ZZZZ).
       The H_edges field is kept at zero for backward compatibility.

    2. **ZZ (Uncertainty-weighted gradient coupling)**:
       −2 × g_strain × √((f×thr_contrast)² + ...) × exp(−((score−thr)/σ)²)
       Ferromagnetic: neighbors ALIGN. g_strain active when Q < 0 (strain).
       The Gaussian uncertainty weight suppresses coupling where the classical
       decision is confident (far from threshold), concentrating quantum
       corrections on the uncertain decision boundary.

    3. **ZZZZ (Circulation plaquette)**: −1 × √((g_rot×f×thr_contrast)² + ...)
       Even-parity: 0/2/4 edges refined. g_rot active when Q > 0 (rotation).
       Threshold-relative contrast on vorticity and current density.

    4. **ZZZZ (X-point plaquette)**: −f_Rm × thr_contrast(max(0, −det(J_B)))
       Even-parity. Detects magnetic reconnection X-points via det(∇B) < 0.
       Orthogonal to K_plaquettes: uses the full Jacobian (sym+antisym),
       not just the curl (antisymmetric) component.
    """

    # Critical thresholds: fixed by physics (NOT trainable)
    # RE_CRIT = 1.0: refine when cell Reynolds Re_cell = v_jump*dx/nu > 1.
    # This means advection dominates diffusion at cell scale — the grid
    # can no longer resolve the physics and refinement is needed.
    # (v8 used RE_CRIT=10 which required extreme under-resolution to trigger,
    # making the Hamiltonian empty for most simulation-resolution grids.)
    #: Percentile du critere RELATIF (voir `_effective_crit`).
    #: Reglage NOUVEAU : il entre dans le perimetre de reoptimisation,
    #: qui passe donc de 8 a 9 parametres.
    RELATIVE_PERCENTILE = 90.0

    RE_CRIT = 1.0      # Cell Reynolds: advection > diffusion at cell scale
    RM_CRIT = 1.0      # Magnetic Reynolds: same criterion for B-field
    MACH_CRIT = 1.0    # Sonic transition
    Q_CRIT = 2.0       # Q-criterion (in std units)
    J_CRIT = 1.0       # Current density threshold

    def __init__(self, cs=1.0, nu=1e-3, eta_mhd=1e-3, dx=1.0,
                 gamma_hydro=0.5, gamma_mag=0.5, kappa=5.0,
                 sigma=0.05, beta_curl=None, beta_xpoint=None,
                 w_z_frac=0.15, relative_percentile=None,
                 beta_grad=None, fixed_curl=True):
        self.cs = cs
        self.nu = nu
        self.eta_mhd = eta_mhd
        self.dx = dx
        # Rotationnel/divergence discrets : voir Simulation.grid. False
        # reproduit bit-a-bit le chemin historique (convention indexing='xy'),
        # True applique la convention AXIS_X/AXIS_Y declaree par le depot.
        self.fixed_curl = bool(fixed_curl)
        # ── Uncertainty width for ZZ coupling ──
        # sigma controls how far from the decision boundary (threshold_amr)
        # the ZZ coupling remains active. Gaussian: exp(-((score-thr)/σ)²)
        # Small σ (0.05): coupling only near boundary → quantum corrections
        #                  where classical decision is uncertain
        # Large σ (0.30): coupling everywhere → reverts to always-on behavior
        self.sigma = sigma

        # Legacy compatibility: if beta_grad is passed (old configs), ignore it
        # and use sigma instead. Log a warning for traceability.
        if beta_grad is not None:
            import warnings
            warnings.warn(
                f"beta_grad={beta_grad} is deprecated; using sigma={sigma} instead. "
                "Update your hyperparams to use 'sigma'.",
                DeprecationWarning, stacklevel=2
            )

        beta = max(beta_curl, beta_xpoint) if beta_curl and beta_xpoint else 1.0
        # ── Michelson sensitivity: split by term type ──
        # If individual betas are not provided, fall back to the shared beta.
        self.beta_curl  = beta_curl  if beta_curl  is not None else beta
        self.beta_xpoint = beta_xpoint if beta_xpoint is not None else beta

        # ── Trainable parameters ──
        self.gamma_hydro = gamma_hydro
        self.gamma_mag = gamma_mag
        self.kappa = kappa
        self.w_z_frac = w_z_frac  # Adaptive Z weight: fraction of max(|C|,|K|)
        # Percentile du critere relatif. Reglable par instance parce qu'il
        # est ENTRAINE (`SEARCH_SPACE`) : c'etait la derniere constante en
        # dur du chemin de decision. `None` retient la constante de classe,
        # donc le comportement d'avant, a l'identique.
        self.relative_percentile = (float(relative_percentile)
                                    if relative_percentile is not None
                                    else self.RELATIVE_PERCENTILE)

    # ══════════════════════════════════════════════════════════════════
    #  f-gate: Normal-Critical scaling (absolute physical non-dimensionalization)
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _f_gate(x, x_crit, gamma, f_max=10.0):
        """
        Normal-critical gate. No sigmoid suppression.

        Normal regime  (x <= x_crit): f = x / x_crit          (linear, proportional)
        Critical regime (x > x_crit): f = 1 + γ × ln(x/x_crit)  (logarithmic growth)

        Continuous at x = x_crit (both sides = 1.0).
        Logarithmic form bounds growth: Re=3000, x_crit=10, γ=2 → f ≈ 12.4
        au lieu de diverger. Cette valeur illustre la FORMULE ; elle ne sort
        pas de la fonction telle qu'elle est appelee, car f_max=10.0 par
        defaut la ramene a 10.0. Le clamp existe pour empecher des
        coefficients extremes de destabiliser le solveur pendant
        l'entrainement.
        """
        r = x / (x_crit + 1e-10)
        f = np.where(r <= 1.0, r, 1.0 + gamma * np.log(np.maximum(r, 1.0)))
        return np.minimum(f, f_max)

    # ══════════════════════════════════════════════════════════════════
    #  g-gates: Leaky sigmoid topological switches
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _g_strain(Q_OW, Q_crit, kappa):
        """
        Leaky sigmoid for strain-dominated regions.

        Convention: grid._compute_q_criterion returns
            Q = 0.5*(omega² - strain²)
        so Q < 0 means strain dominates, Q > 0 means rotation dominates.

        g_strain activates when Q << 0 (strain):
            g = 1 / (1 + exp(+κ × Q / Q_crit))
        """
        x = np.clip(kappa * Q_OW / (Q_crit + 1e-10), -500, 500)
        return 1.0 / (1.0 + np.exp(x))

    @staticmethod
    def _g_rot(Q_OW, Q_crit, kappa):
        """
        Leaky sigmoid for rotation-dominated regions.

        Convention: grid._compute_q_criterion returns
            Q = 0.5*(omega² - strain²)
        so Q > 0 means rotation dominates.

        g_rot activates when Q >> 0 (rotation):
            g = 1 / (1 + exp(−κ × Q / Q_crit))
        """
        x = np.clip(-kappa * Q_OW / (Q_crit + 1e-10), -500, 500)
        return 1.0 / (1.0 + np.exp(x))

    @staticmethod
    def _g_mag(Jz_curl, J_crit, kappa):
        """
        Leaky sigmoid for magnetically active regions (|Jz| > J_crit).

        g_mag(J) = 1 / (1 + exp(−κ × (|Jz|/J_crit − 1)))

        When |Jz| >> J_crit: g → 1.0 (current sheet active)
        When |Jz| << J_crit: g → 0 (weak current, suppressed)
        """
        x = np.clip(-kappa * (np.abs(Jz_curl) / (J_crit + 1e-10) - 1.0), -500, 500)
        return 1.0 / (1.0 + np.exp(x))

    def _effective_crit(self, signal, crit_absolu):
        """Seuil effectif : `min(absolu, percentile)`.

        Le seuil de maille est ABSOLU : `RE_CRIT nu/(dx^2 v0)`. Il croit
        donc en 1/dx^2, et il est le MEME pour tous les scenarios. Deux
        consequences mesurees :

          - il meurt au raffinement. Sur un champ physique fixe (rotation
            solide), `K_plaquettes` passe de 1.00e+02 a N=32 a EXACTEMENT
            0 a N=256, la resolution d'entrainement.

          - il ne peut pas servir deux instabilites d'amplitudes
            differentes. |omega| vaut au maximum 1.55e-02 sur
            harris_tearing et 1.96e+01 sur mhd_rotor : trois ordres de
            grandeur, pour un seuil unique de 13.04.

        Or l'information EST la. Contraste max/mediane du signal brut a
        N=256 : 1104 sur harris_tearing (sqrt(det)), 223 sur
        island_coalescence, 752 sur mhd_rotor. Ce n'est pas la structure
        qui manque, c'est le seuil absolu qui l'efface.

        D'ou `min(absolu, percentile)` :

          - des qu'une cellule franchit le critere physique, l'absolu
            l'emporte et le comportement d'origine est conserve A
            L'IDENTIQUE ;
          - sinon le relatif prend le relais et distingue les cellules les
            plus instables DU CHAMP COURANT.

        Un champ rigoureusement uniforme n'a pas de cellule « plus
        instable » : son percentile vaut son maximum, le contraste seuille
        rend zero partout. Le critere ne fabrique donc pas de signal a
        partir de rien -- c'est l'invariant que teste
        `test_le_critere_relatif_ne_fabrique_pas_de_signal`.
        """
        fini = np.asarray(signal, dtype=float)
        fini = fini[np.isfinite(fini)]
        if fini.size == 0:
            return crit_absolu
        if float(fini.max()) >= crit_absolu:
            return crit_absolu          # le critere physique tire deja
        return float(np.percentile(fini, self.relative_percentile))

    @staticmethod
    def _compute_det_jacobian_B(Bx, By, dx):
        """
        Determinant of the magnetic Jacobian: det(∇B).

        det(J_B) = ∂Bx/∂x × ∂By/∂y − ∂Bx/∂y × ∂By/∂x

        Topological discriminant for magnetic null points:
          det < 0 → X-point (hyperbolic null, reconnection site)
          det > 0 → O-point (elliptic null, island center)
          det ≈ 0 → away from nulls (quiet region)

        Uses central differences for consistency with Q-criterion.
        """
        dBx_dx = 0.5 * (np.roll(Bx, -1, axis=0) - np.roll(Bx, 1, axis=0)) / dx
        dBx_dy = 0.5 * (np.roll(Bx, -1, axis=1) - np.roll(Bx, 1, axis=1)) / dx
        dBy_dx = 0.5 * (np.roll(By, -1, axis=0) - np.roll(By, 1, axis=0)) / dx
        dBy_dy = 0.5 * (np.roll(By, -1, axis=1) - np.roll(By, 1, axis=1)) / dx
        return dBx_dx * dBy_dy - dBx_dy * dBy_dx

    # ══════════════════════════════════════════════════════════════════
    #  Michelson contrast filter (legacy, kept for reference)
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _michelson_relu(val, avg, beta):
        """
        Michelson contrast filter + ReLU.

        Mic(val, avg) = max(0, (β×val − avg) / (β×val + avg + ε))

        Local spatial normalization (NOT global domain normalization).
        Bounded [0, 1). If the entire domain is uniform, Mic → 0.
        """
        num = beta * val - avg
        den = beta * val + avg
        return np.maximum(0, num / (den + 1e-10))

    # ══════════════════════════════════════════════════════════════════
    #  Threshold-relative contrast (replaces Michelson for ZZ/ZZZZ)
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _threshold_contrast(val, val_crit, beta, tc_max=10.0):
        """
        Threshold-relative contrast filter.

        signal = β × max(0, val/val_crit − 1)

        Unlike Michelson which compares val to its spatial average
        (killing the signal when the domain is uniformly active),
        this compares val to a fixed physical threshold.

        Clamped to tc_max to prevent extreme coefficient build-up
        that can destabilise the solver via degenerate QAOA decisions.
        """
        raw = beta * np.maximum(0.0, val / (val_crit + 1e-10) - 1.0)
        return np.minimum(raw, tc_max)

    # ══════════════════════════════════════════════════════════════════
    #  Physical score — NOT wired to θ initialization in the deployed
    #  pipeline. See the docstring below (D-176, RESULTS.md) before
    #  reading this header as a claim about production behaviour.
    # ══════════════════════════════════════════════════════════════════

    LOHNER_CRIT = 0.3   # Löhner > 0.3 → genuine discontinuity

    def physical_score(self, physics_state):
        """
        Physics-grounded instability score — an alternative to
        `AngleMapper.classical_score`, exercised only by the test suite.

        D-176 : cette docstring annoncait "replaces classical_score for
        theta initialization" et se presentait comme le score DEPLOYE.
        C'etait deja faux avant D-9 (qui a corrige le seul appelant errone,
        `h3_uncertainty_window.py`, sans toucher a cette ligne) et c'est
        toujours faux : AUCUN site de `src/` ni de `study/` n'appelle
        `physical_score` (verifie par grep, 0 site hors sa propre
        definition et ses tests). Le theta-init deploye vient partout de
        `AngleMapper.classical_score` (`refinement.py`, `qaoa_inputs.py`).
        `physical_score` n'est ni mort au sens du bytecode -- ~30 sites de
        tests l'appellent directement, comme formule alternative a
        comparer -- ni vivant au sens du pipeline : aucun artefact publie
        n'en depend. Hors chemin critique, comportement inchange : une
        ligne dans `RESULTS.md`, pas d'entree `DEFAUTS.md`.

        Each indicator is normalized by its **physical critical value**
        (not by the domain maximum). This ensures:
        - Noise / quiet regions → score ≈ 0  (no refinement)
        - Physically dangerous features → score → 1 (refine)

        Indicators and critical scales:
        --------------------------------
        1. Vorticity |ωz|: critical when cell vortex Reynolds
           Re_ω = |ωz|·dx²/ν > RE_CRIT.
           Finite-diff critical value: RE_CRIT × ν / dx.

        2. Divergence |∇·v|: critical when compression rate
           approaches sound speed. Finite-diff critical value: c_s.

        3. Current density |Jz|: critical when Lundquist number
           S = |Jz|·dx²/η > RM_CRIT.
           Critical value: RM_CRIT × η / dx².

        4. Löhner estimator on |B|: already dimensionless.
           Critical value: LOHNER_CRIT = 0.3.

        Returns (N, N) array in [0, 1].
        """
        vx, vy = physics_state['vx'], physics_state['vy']
        Bx, By = physics_state['Bx'], physics_state['By']
        Jz = physics_state['Jz']

        # ── Physical critical values for finite-difference quantities ──
        omega_crit = self.RE_CRIT * self.nu / self.dx
        div_crit = self.cs
        jz_crit = self.RM_CRIT * self.eta_mhd / (self.dx ** 2)

        # ── Raw indicators (same discrete operators as classical_score) ──
        vorticity = np.abs(curl_z(vx, vy, self.fixed_curl))
        div_v = np.abs(divergence(vx, vy, self.fixed_curl))
        abs_Jz = np.abs(Jz)
        B_mag = np.sqrt(Bx**2 + By**2)
        lohner_B = _lohner_estimator(B_mag)

        # ── Normalize by physical critical values, clip at 1 ──
        s_vort = np.clip(vorticity / max(omega_crit, 1e-10), 0.0, 1.0)
        s_div  = np.clip(div_v / max(div_crit, 1e-10), 0.0, 1.0)
        s_jz   = np.clip(abs_Jz / max(jz_crit, 1e-10), 0.0, 1.0)
        s_loh  = np.clip(lohner_B / self.LOHNER_CRIT, 0.0, 1.0)

        score = np.sqrt((s_vort**2 + s_div**2 + s_jz**2 + s_loh**2) / 4.0)
        return np.clip(score, 0.0, 1.0)

    # ══════════════════════════════════════════════════════════════════
    #  Main computation
    # ══════════════════════════════════════════════════════════════════

    def compute_coefficients(self, sim, score,
                              fields, threshold_amr,
                              advanced_anomalies_enabled=False,
                              dx_override=None,
                              **kwargs):
        """
        Compute Hamiltonian coefficients (Z + ZZ + ZZZZ).

        Parameters
        ----------
        sim              : MHDSolver instance (grid methods used for gradients).
        score            : (N, N) pre-computed classical instability score.
        fields           : dict with 'vx', 'vy', 'Bx', 'By', 'Jz'.
        threshold_amr    : threshold (used only for classical AMR comparison).
        advanced_anomalies_enabled : whether to compute X-point reconnection terms.
        dx_override      : float, optional.  When computing on downsampled
                           VQA-resolution fields, pass the effective cell size
                           (patch_physical_size / target_dim).  Overrides
                           self.dx for threshold computations, gradient
                           operators, and the f-gate / threshold-contrast
                           critical values.

        Returns
        -------
        dict with 'H_edges', 'C_edges', 'K_plaquettes', ['K_xpoint']

        Notes
        -----
        H_edges uses an adaptive Z bias: α × (score − threshold_amr),
        where α = w_z_frac × median(nonzero |C|, |K|). The median is
        robust to outlier cells (e.g. a single cell with C=-1193) while
        reflecting the typical coupling scale. This breaks the degenerate
        ground state of the ferromagnetic ZZ/ZZZZ Hamiltonian while
        keeping the Z weight small enough for ZZ correlations to dominate.
        """
        dx = dx_override if dx_override is not None else self.dx

        vx, vy = fields['vx'], fields['vy']
        Bx, By = fields['Bx'], fields['By']
        Jz = fields['Jz']

        # ── Shared physics fields ─────────────────────────────────────
        # Okubo-Weiss Q-criterion (Q > 0 = rotation, Q < 0 = strain)
        Q_OW = sim.grid._compute_q_criterion(vx, vy, dx=dx)

        # Reference scales for non-dimensionalization
        v0 = max(np.mean(np.abs(vx)), np.mean(np.abs(vy)), 1e-10)
        B0 = max(np.mean(np.abs(Bx)), np.mean(np.abs(By)), 1e-10)

        # Discrete vorticity and current density (for plaquette)
        omega_z = curl_z(vx, vy, self.fixed_curl)
        Jz_curl = curl_z(Bx, By, self.fixed_curl)

        # ── 0. ACTIVITY BIAS (1-body Z) ── ADAPTIVE WEIGHT ─────────
        # Z bias breaks the degenerate ground state of ferromagnetic ZZ/ZZZZ.
        # Without Z, both all-|0⟩ and all-|1⟩ minimize the energy equally,
        # and COBYLA can converge to all-|0⟩ (suppress ALL refinement).
        #
        # The weight is adaptive: α = w_z_frac × max(|C|, |K|).
        # This ensures Z is always a FRACTION of the ZZ/ZZZZ scale:
        #   - Small enough that ZZ correlations still dominate (quantum advantage)
        #   - Large enough to break degeneracy (prevents all-|0⟩ collapse)
        #
        # H = α × (score − threshold_amr):
        #   score > threshold → H > 0 → bias toward |1⟩ (refine)
        #   score < threshold → H < 0 → bias toward |0⟩ (don't refine)
        #   score = threshold → H = 0 → no bias at decision boundary
        #     (ZZ corrections matter most here)
        #
        # The Z bias is computed AFTER C_edges and K_plaquettes so that
        # the adaptive scale is known. Placeholder zeros are set here
        # and filled in after ZZ/ZZZZ computation.
        N_field = score.shape[0]
        H_horiz = np.zeros((N_field, N_field))
        H_vert = np.zeros((N_field, N_field))

        # ── 1. GRADIENT COUPLING (2-body ZZ) ─────────────────────────
        # C = 2 × g_strain(Q_OW) × √((f_hydro(Re) × Mic(Δv))² + (f_mag(Rm) × Mic(ΔB))²)
        # Sign: ferromagnetic (−C × Z_i Z_j) → neighbors ALIGN.
        # We output −C so cost_hamiltonian uses the value as-is.
        v_jump_h = sim.grid._get_vector_jump(vx, vy, axis=1)
        B_jump_h = sim.grid._get_vector_jump(Bx, By, axis=1)
        v_jump_v = sim.grid._get_vector_jump(vx, vy, axis=0)
        B_jump_v = sim.grid._get_vector_jump(Bx, By, axis=0)

        # Cell Reynolds / Magnetic Reynolds numbers
        Re_h = (v_jump_h * dx) / max(self.nu, 1e-10)
        Rm_h = (B_jump_h * dx) / max(self.eta_mhd, 1e-10)
        Re_v = (v_jump_v * dx) / max(self.nu, 1e-10)
        Rm_v = (B_jump_v * dx) / max(self.eta_mhd, 1e-10)

        # f-gates: normal-critical scaling
        f_Re_h = self._f_gate(Re_h, self.RE_CRIT, self.gamma_hydro)
        f_Rm_h = self._f_gate(Rm_h, self.RM_CRIT, self.gamma_mag)
        f_Re_v = self._f_gate(Re_v, self.RE_CRIT, self.gamma_hydro)
        f_Rm_v = self._f_gate(Rm_v, self.RM_CRIT, self.gamma_mag)

        # g-gate: strain topology (Q < 0 activates gradient detection)
        g_strain = self._g_strain(Q_OW, self.Q_CRIT, self.kappa)

        # Threshold-relative contrast on raw jumps (replaces Michelson).
        # Critical values: a jump is significant when it produces a cell
        # Reynolds/Rm above the critical threshold. The critical jump is:
        #   v_jump_crit = RE_CRIT * nu / dx,  B_jump_crit = RM_CRIT * eta / dx
        # beta=1.0 (fixed): threshold contrast sensitivity for gradient terms
        # is no longer a trainable parameter — replaced by sigma (uncertainty width).
        BETA_GRAD_FIXED = 1.0
        v_jump_crit = self.RE_CRIT * self.nu / max(dx, 1e-10)
        B_jump_crit = self.RM_CRIT * self.eta_mhd / max(dx, 1e-10)
        mic_v_h = self._threshold_contrast(v_jump_h, v_jump_crit, BETA_GRAD_FIXED)
        mic_v_v = self._threshold_contrast(v_jump_v, v_jump_crit, BETA_GRAD_FIXED)
        mic_B_h = self._threshold_contrast(B_jump_h, B_jump_crit, BETA_GRAD_FIXED)
        mic_B_v = self._threshold_contrast(B_jump_v, B_jump_crit, BETA_GRAD_FIXED)

        # Ferromagnetic: output negative so cost_hamiltonian uses as-is
        C_horiz = -2.0 * g_strain * np.sqrt(
            (f_Re_h * mic_v_h)**2 + (f_Rm_h * mic_B_h)**2
        )
        C_vert = -2.0 * g_strain * np.sqrt(
            (f_Re_v * mic_v_v)**2 + (f_Rm_v * mic_B_v)**2
        )

        # ── 2nd-order gradient (curvature contrast) ──
        # Captures difference in Laplacian between neighbors.
        # Distinguishes genuine discontinuities (large curvature contrast)
        # from smooth ramps (small curvature contrast). This gives the QAOA
        # additional spatial correlation information beyond 1st-order jumps.
        d2_v_h = sim.grid._get_second_order_jump(vx, vy, axis=1, dx=dx)
        d2_B_h = sim.grid._get_second_order_jump(Bx, By, axis=1, dx=dx)
        d2_v_v = sim.grid._get_second_order_jump(vx, vy, axis=0, dx=dx)
        d2_B_v = sim.grid._get_second_order_jump(Bx, By, axis=0, dx=dx)

        # Threshold-relative contrast on 2nd-order jumps.
        # Critical curvature: same scale as 1st-order jump / dx
        d2_v_crit = v_jump_crit / max(dx, 1e-10)
        d2_B_crit = B_jump_crit / max(dx, 1e-10)
        mic_d2v_h = self._threshold_contrast(d2_v_h, d2_v_crit, BETA_GRAD_FIXED)
        mic_d2v_v = self._threshold_contrast(d2_v_v, d2_v_crit, BETA_GRAD_FIXED)
        mic_d2B_h = self._threshold_contrast(d2_B_h, d2_B_crit, BETA_GRAD_FIXED)
        mic_d2B_v = self._threshold_contrast(d2_B_v, d2_B_crit, BETA_GRAD_FIXED)

        # Add to gradient coupling (ferromagnetic, same sign convention)
        C_horiz += -1.0 * g_strain * np.sqrt(
            (f_Re_h * mic_d2v_h)**2 + (f_Rm_h * mic_d2B_h)**2
        )
        C_vert += -1.0 * g_strain * np.sqrt(
            (f_Re_v * mic_d2v_v)**2 + (f_Rm_v * mic_d2B_v)**2
        )

        # ── Uncertainty weighting ──
        # Modulate ZZ coupling by Gaussian centered at threshold_amr:
        #   w(score) = exp(-((score - threshold_amr) / sigma)²)
        #
        # This concentrates coupling near the decision boundary where the
        # classical score is uncertain (score ≈ threshold). Far from
        # threshold, the classical decision is already confident and ZZ
        # coupling would just redundantly enforce agreement.
        #
        # The weight is computed per-cell using the average score of the
        # two neighbors connected by each edge.
        #
        # Note: score may include halo padding (e.g. 6×6) while fields
        # (and thus C arrays) are at a different resolution (e.g. 4×4).
        # Resize score to match the field grid before computing weights.
        sigma = max(self.sigma, 1e-6)
        field_shape = vx.shape
        if score.shape != field_shape:
            from scipy.ndimage import zoom
            score_resized = zoom(score, (field_shape[0] / score.shape[0],
                                         field_shape[1] / score.shape[1]), order=1)
        else:
            score_resized = score
        # Horizontal edges connect (i,j) and (i,j+1)
        score_avg_h = 0.5 * (score_resized + np.roll(score_resized, -1, axis=1))
        uncertainty_h = np.exp(-((score_avg_h - threshold_amr) / sigma) ** 2)
        # Vertical edges connect (i,j) and (i+1,j)
        score_avg_v = 0.5 * (score_resized + np.roll(score_resized, -1, axis=0))
        uncertainty_v = np.exp(-((score_avg_v - threshold_amr) / sigma) ** 2)

        C_horiz *= uncertainty_h
        C_vert *= uncertainty_v

        # ── 2. CIRCULATION PLAQUETTE (4-body ZZZZ) ───────────────────
        # K = g_rot × Mic(circulation) + g_mag × Mic(|Jz|)
        # Sign: even-parity (−K × ZZZZ) → 0/2/4 edges refined.

        # g-gates: decoupled topological switches
        g_rot = self._g_rot(Q_OW, self.Q_CRIT, self.kappa)
        # `Jz_curl` sort de `curl_z`, qui ne divise PAS par dx : c'est une
        # difference finie en unites de GRILLE. `Q_OW`, lui, vient de
        # `_compute_q_criterion(..., dx=dx)` et est en unites PHYSIQUES.
        #
        # Les deux portes topologiques comparaient donc des grandeurs de
        # deux systemes d'unites differents a des seuils de meme ordre
        # nominal (Q_CRIT = 2.0, J_CRIT = 1.0). La porte magnetique etait
        # plus dure a franchir d'un facteur exactement 1/dx -- 10.2 a
        # N=64, 20.4 a N=128, 40.7 a N=256 : elle se degradait quand la
        # grille se raffine.
        #
        # Mesure avant, nappe de courant a N=64 : mic_jz = 1.541e-01 et
        # f_Rm_cell = 8.346 (les deux sains), mais g_mag = 0.000, d'ou
        # mag_comp = 1.816e-05 contre fluid_comp = 5.009e-01 sur un
        # reseau de vortex -- un facteur 27 500 entre deux instabilites de
        # meme nature.
        #
        # `g_mag` recoit desormais un Jz PHYSIQUE, comme `g_rot` recoit un
        # Q_OW physique. Voir RESULTS.md.
        Jz_phys = Jz_curl / max(dx, 1e-10)
        g_mag = self._g_mag(Jz_phys, self.J_CRIT, self.kappa)

        # f-gates for plaquette: Re for fluid component, Rm for magnetic
        Re_cell = (np.sqrt(vx**2 + vy**2) * dx) / max(self.nu, 1e-10)
        Rm_cell = (np.sqrt(Bx**2 + By**2) * dx) / max(self.eta_mhd, 1e-10)
        f_Re_cell = self._f_gate(Re_cell, self.RE_CRIT, self.gamma_hydro)
        f_Rm_cell = self._f_gate(Rm_cell, self.RM_CRIT, self.gamma_mag)

        # Threshold-relative contrast on vorticity and current density.
        # Critical vorticity: Re_omega = |omega|*dx^2/nu > RE_CRIT
        #   → omega_crit = RE_CRIT * nu / dx^2  (but we use omega/v0, so divide by v0)
        # Critical Jz: Rm_J = |Jz|*dx^2/eta > RM_CRIT
        #   → jz_crit = RM_CRIT * eta / dx^2  (but we use Jz/B0, so divide by B0)
        omega_mag = np.abs(omega_z / v0)
        jz_mag = np.abs(Jz_curl / B0)
        omega_crit = self.RE_CRIT * self.nu / (max(dx, 1e-10)**2 * max(v0, 1e-10))
        jz_crit = self.RM_CRIT * self.eta_mhd / (max(dx, 1e-10)**2 * max(B0, 1e-10))
        # Critere RELATIF : voir `_effective_crit`.
        omega_crit_eff = self._effective_crit(omega_mag, omega_crit)
        jz_crit_eff = self._effective_crit(jz_mag, jz_crit)
        mic_omega = self._threshold_contrast(omega_mag, omega_crit_eff, self.beta_curl)
        mic_jz = self._threshold_contrast(jz_mag, jz_crit_eff, self.beta_curl)

        # Decoupled components with Michelson normalization
        fluid_comp = g_rot * f_Re_cell * mic_omega
        mag_comp = g_mag * f_Rm_cell * mic_jz

        # Even-parity: output negative so cost_hamiltonian uses as-is
        K_plaquettes = -1.0 * np.sqrt(fluid_comp**2 + mag_comp**2)

        # ── Etages observables ─────────────────────────────────────
        # Le desequilibre entre canal fluide et canal magnetique ne peut
        # pas etre diagnostique en RECALCULANT les etages a cote : trois
        # fois de suite, une reproduction incomplete a fait accuser du code
        # juste. On expose donc les composantes telles que la fonction les
        # a calculees. `_stages` n'est lu que par les tests et les
        # diagnostics ; aucun chemin de production ne le consulte.
        self._stages = {
            "g_rot": g_rot, "g_mag": g_mag,
            "f_Re_cell": f_Re_cell, "f_Rm_cell": f_Rm_cell,
            "mic_omega": mic_omega, "mic_jz": mic_jz,
            "omega_mag": omega_mag, "jz_mag": jz_mag,
            "omega_crit": omega_crit_eff, "jz_crit": jz_crit_eff,
            "fluid_comp": fluid_comp, "mag_comp": mag_comp,
            "v0": v0, "B0": B0,
        }

        # ── Build result ───────────────────────────────────────────
        # ── 0b. FILL Z BIAS with adaptive weight (global median) ────
        # The Z bias breaks the degenerate ground state of ferromagnetic
        # ZZ/ZZZZ. We scale it relative to a GLOBAL summary of the
        # multi-body coupling strength.
        #
        # Using the global MAX caused a single outlier (e.g. C=-1193 at
        # the Orszag-Tang current sheet) to over-inflate alpha for all
        # cells. The MEDIAN of non-zero |C| and |K| values is robust to
        # outliers while still reflecting the typical coupling scale.
        # w_z_frac ∈ [0.05, 0.5], default 0.15.
        all_coeffs = np.concatenate([
            np.abs(C_horiz).ravel(),
            np.abs(C_vert).ravel(),
            np.abs(K_plaquettes).ravel(),
        ])
        nonzero = all_coeffs[all_coeffs > 1e-10]
        C_scale = float(np.median(nonzero)) if len(nonzero) > 0 else 0.0
        alpha_z = self.w_z_frac * C_scale
        z_bias = alpha_z * (score - threshold_amr)
        H_horiz = z_bias
        H_vert = z_bias

        result = {
            "H_edges": (H_horiz, H_vert),
            "C_edges": (C_horiz, C_vert),
            "K_plaquettes": K_plaquettes,
            "threshold_amr": threshold_amr,
            "w_z_frac": self.w_z_frac,
        }

        # ── 3. X-POINT RECONNECTION (4-body ZZZZ, same plaquette topology) ─
        # Detects magnetic reconnection X-points via det(J_B) < 0.
        # Orthogonal to K_plaquettes: uses the full Jacobian determinant
        # (symmetric + antisymmetric derivatives), not just |Jz| (curl only).
        #
        # Signal: max(0, −det(J_B))  — auto-limiting:
        #   X-point (hyperbolic null): det < 0 → strong signal
        #   O-point (elliptic null):   det > 0 → zero signal
        #   Away from nulls:           det ≈ 0 → zero signal
        #
        # No separate g-gate needed (signal is intrinsically localized).
        if advanced_anomalies_enabled:
            det_J_B = self._compute_det_jacobian_B(Bx, By, dx)

            # X-point signal: only negative determinant (hyperbolic nulls)
            xpoint_signal = np.maximum(0.0, -det_J_B)

            # `sqrt(det)` a les MEMES UNITES que |Jz| : tous deux sont des
            # gradients de B. On emploie donc la normalisation et le seuil du
            # canal courant, deja definis plus haut (`jz_crit`).
            #
            # L'ancienne forme comparait `sig / (B0/dx)^2` a
            # `(RM_CRIT eta / (dx B0))^2`. `sig` est un gradient AU CARRE
            # normalise par une seule puissance de dx^2, puis compare a un
            # seuil lui-meme au carre : le rapport variait en **dx^4**. Le
            # critere devenait donc moins susceptible de se declencher a
            # mesure que la grille se raffine, a la puissance quatre.
            #
            # Mesure du rapport signal/seuil (il faut depasser 1 pour tirer) :
            #
            #                        N=64     N=128    N=256   loi
            #   ancienne forme      0.171    0.0105   0.0007   dx^4
            #   forme actuelle      0.414    0.1025   0.0256   dx^2
            #   |Jz|, reference     5.137    1.349    0.341    dx^2
            #
            # La forme actuelle suit la meme loi que le canal courant, ce que
            # la coherence dimensionnelle impose. Voir RESULTS.md.
            xpoint_grad = np.sqrt(xpoint_signal) / max(B0, 1e-10)

            # Meme critere relatif, mais sur la distribution DE CE CANAL :
            # `sqrt(det)` et |Jz| n'ont pas les memes percentiles, et c'est
            # precisement ce qui rend le canal point X discriminant la ou
            # le canal courant ne l'est pas (contraste 1104 contre 49.5 sur
            # harris_tearing).
            xpoint_crit_eff = self._effective_crit(xpoint_grad, jz_crit)
            mic_xpoint = self._threshold_contrast(
                xpoint_grad, xpoint_crit_eff, self.beta_xpoint
            )

            # PAS de `f_Rm_cell` ici : cette porte vaut
            # `_f_gate(|B| dx / eta)` et un point X est PAR DEFINITION un
            # zero de B. Elle annulait donc le coefficient exactement a
            # l'endroit qu'il doit signaler, ne laissant que l'anneau
            # autour. Mesure sur une nappe de 2 cellules a N=256 : seuil au
            # point X = 0.5292, porte au point X = 0.0000 (sur les six
            # epaisseurs testees), coefficient resultant 0.0000 contre
            # 0.8537 sur l'anneau.
            #
            # Le commentaire d'origine annoncait deja « No separate g-gate
            # needed (signal is intrinsically localized) » pendant que le
            # code appliquait la porte : le commentaire et le code se
            # contredisaient. Voir tests/mapping/test_xpoint_at_training_resolution.py
            #
            # ATTENTION — ce retrait ne suffit pas a faire tirer le terme sur
            # des champs REELS : a N=256 le seuil lui-meme n'est pas atteint
            # (signal/seuil = 7e-4 sur island_coalescence). La normalisation
            # est le second verrou, et il reste ouvert. Voir RESULTS.md.
            #
            # Even-parity ZZZZ: output negative so cost_hamiltonian uses as-is
            K_xpoint = -1.0 * mic_xpoint
            result["K_xpoint"] = K_xpoint

        return result