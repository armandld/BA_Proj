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

  ZZZZ scindee, un terme par TYPE de structure :
    K_vort = -w_ZZZZ * |omega_z| / max|omega_z|
    K_curr = -w_ZZZZ * |J_z|     / max|J_z|

    `K_plaquettes` somme |omega| et |J| : un vortex pur et une nappe de
    courant pure y rendent EXACTEMENT la meme valeur, donc le terme capte
    « il se passe quelque chose », pas un type. Les deux termes ci-dessus
    separent les deux, chacun normalise par le max de SON signal — dans les
    deux modes, y compris `legacy`, car les normaliser par une moyenne
    reintroduirait le couplage entre spikiness du champ et poids des
    familles.

    OPT-IN : `split_plaquette=True`. Hors du defaut a dessein. Le
    dictionnaire rendu est consomme comme un TOUT par `call_vqa_shell.py`,
    qui somme |coeff| sur toutes les cles tableau pour former `E_max` :
    deux cles de plus le deplacent de +15,9 % (`legacy`) / +34,2 % (`max`)
    alors qu'aucune valeur partagee ne bouge. Verifier l'egalite cle par
    cle ne suffit donc PAS a prouver qu'un consommateur ne bouge pas ;
    c'est l'ENSEMBLE des cles qui fait partie du contrat.

    `K_plaquettes` reste inchange bit a bit dans les deux modes, et le
    circuit ne lit pas encore les deux termes scindes.

  ZZZZ X-point (optional):
    K_xp = -w_ZZZZ * max(0, -det(nabla B)) / (|det(nabla B)|_max + eps)

  Z (bias):
    h_i = +c * median(|C|, |K|) * (s_i - thr)
    where c = 0.1 fixed
    Le signe est POSITIF (cf. ligne du calcul de z_bias) : au-dessus du
    seuil le biais pousse vers |1> (raffiner). Cette ligne portait un
    signe negatif, contraire au code ; seule la documentation etait fausse.

Fixed weights: w_ZZ = 2, w_ZZZZ = 1. Negative signs = ferromagnetic.

Normalisation (`norm=`, defaut `legacy`)
----------------------------------------
Les formules ci-dessus sont le chemin `legacy`. Elles emploient TROIS
normalisateurs differents — moyenne (ZZ), somme de deux maxima (ZZZZ),
mediane (biais) — sur un champ dont la FORME depend de `dim` : moyenner par
blocs a la resolution `dim` est un filtre passe-bas, et monter `dim` resout
des echelles plus fines, donc dissymetrise la distribution des sauts. Le pic
suit la queue, la mediane non : le rapport biais/couplage DERIVE avec `dim`,
par construction. Un reglage obtenu a une taille ne transfere alors pas a
une autre, et un balayage en `dim` mesure deux choses a la fois.

`norm="max"` accroche les trois termes a la meme statistique :

    C_ij = -w_ZZ   * |saut_ij|      / max|saut|          -> max|C| == w_ZZ
    K_p  = -w_ZZZZ * (|omega|+|J|)  / max(|omega|+|J|)   -> max|K| == w_ZZZZ
    h_i  = +c_bias * max(|C|,|K|)   * (s_i - thr)

L'EQUILIBRE entre les termes devient independant de `dim` (rapport fixe a
`c_bias`), et les gardes y sont multiplicatifs, si bien que les invariances
en `dx` et en amplitude sont EXACTES (4,8e-16) la ou `legacy` ne les tient
qu'a 9,8e-11 pres — son `+ EPS` additif decale l'echelle.

Ce que `max` ne fait PAS : rendre les coefficients identiques d'un `dim` a
l'autre. Le MOTIF spatial change necessairement, puisque le champ d'entree
change avec la coupure du filtre.

Ce qui n'est PAS un argument contre `max`, contrairement a ce qu'une
version anterieure de ce texte laissait entendre : sous `legacy`, le fait
que max|C| varie avec le champ n'est pas une « information de structure »
utile. C'est le POIDS RELATIF de ZZ contre ZZZZ qui varie — mesure, il passe
de 3,12 a 8,09 entre une rotation lisse et une nappe raide, un facteur 2,6
sur l'equilibre de deux familles de termes decide par la seule spikiness de
l'entree au lieu de la conception. Sous `max` ce rapport vaut w_zz/w_zzzz
partout. Il n'y a donc pas d'arbitrage : `max` retire un couplage parasite.

`tests/mapping/test_normalisation_max_invariante.py` mesure les deux.

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
  - Added: simple domain-normalized ratios (voir « Normalisation »)
  - Result: 0 trainable parameters in Hamiltonian (was ~8)
"""

import numpy as np

from Simulation.grid import curl_z


class PhysicalMapperV2:
    """
    Parameter-free Hamiltonian coefficient computation from MHD fields.

    Seul `thr_amr` (et les poids fixes w_zz / w_zzzz / c_bias) change la
    sortie. Ce mappeur est ADIMENSIONNEL : chaque terme est divise par une
    norme prise sur le meme champ, si bien que

      - `dx` se simplifie exactement (il n'apparait que dans
        `_compute_det_jacobian_B`, ou det ∝ 1/dx² est ensuite divise par
        max|det| ∝ 1/dx²) ;
      - l'amplitude des champs se simplifie de meme (multiplier v et B par
        10 laisse C, K et H inchanges) ;
      - `nu` et `eta` n'apparaissent nulle part : aucun nombre de Reynolds
        n'entre dans le v2.

    Consequence a garder en tete : le v2 ne peut pas distinguer un ecoulement
    visqueux d'un ecoulement inertiel, ni une grille fine d'une grille
    grossiere. Il ne voit que la FORME relative des champs. Un transfert
    entre nombres de Reynolds est donc trivialement satisfait par les
    coefficients — la dependance en Re ne peut venir que du score externe.
    `tests/test_mapper_contracts.py` mesure ces trois invariances.
    """

    # Fixed weights (not trained, chosen once by physical reasoning)
    W_ZZ = 2.0       # ZZ coupling weight
    W_ZZZZ = 1.0     # ZZZZ coupling weight
    C_BIAS = 0.1     # Z bias scale: fraction of median(|C|,|K|). Default.
    EPS = 1e-10       # division-by-zero guard

    #: Normalisations disponibles. `legacy` = le chemin historique (moyenne
    #: pour ZZ, somme des deux maxima pour ZZZZ, mediane pour le biais).
    #: `max` = tout normalise par le max de son propre signal, biais accroche
    #: au max des couplages.
    NORMALISATIONS = ("legacy", "max")

    def __init__(self, dx=1.0, c_bias=None, w_zz=None, w_zzzz=None,
                 fixed_curl=True, norm="legacy"):
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
        # Voir Simulation.grid : False reproduit bit-a-bit le chemin
        # historique, True applique la convention AXIS_X/AXIS_Y du depot.
        self.fixed_curl = bool(fixed_curl)
        if norm not in self.NORMALISATIONS:
            raise ValueError(
                f"norm={norm!r} inconnue ; attendu l'une de "
                f"{self.NORMALISATIONS}")
        self.norm = norm
        self.c_bias = self.C_BIAS if c_bias is None else float(c_bias)
        self.w_zz = self.W_ZZ if w_zz is None else float(w_zz)
        self.w_zzzz = self.W_ZZZZ if w_zzzz is None else float(w_zzzz)

    # ------------------------------------------------------------------
    #  Main computation
    # ------------------------------------------------------------------

    def compute_coefficients(self, sim, score, fields, threshold_amr,
                             advanced_anomalies_enabled=False,
                             dx_override=None, split_plaquette=False,
                             **kwargs):
        """
        Compute Hamiltonian coefficients (Z + ZZ + ZZZZ).

        Parameters
        ----------
        sim : MHDSolver or None
            Non utilise. Le v2 reimplemente ses operateurs de saut en ligne
            (voir plus bas) au lieu d'appeler `sim.grid._get_vector_jump`.
            L'argument est conserve pour garder la signature commune avec
            `PhysicalMapper.compute_coefficients` (v1), que le pipeline
            appelle sans savoir lequel des deux il tient.
            `tests/test_mapper_contracts.py` verifie que les deux
            implementations coincident encore.
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
        split_plaquette : bool, default False
            Ajoute `K_vorticity` et `K_current`, la plaquette SCINDEE par
            type de structure. **Hors du defaut a dessein** : le dictionnaire
            rendu est consomme comme un TOUT par `src/call_vqa_shell.py`, qui
            somme `|coeff|` sur toutes les cles tableau pour former `E_max`.
            Deux cles de plus deplacent donc `E_max` de +15,9 % (`legacy`) et
            +34,2 % (`max`) — mesure — sans que la moindre valeur partagee
            change. Le circuit ne lit pas encore les deux termes ; les rendre
            par defaut serait un changement de comportement de `src/` fait
            « au passage ».

        Returns
        -------
        dict with 'H_edges', 'C_edges', 'K_plaquettes', ['K_xpoint'],
             'threshold_amr', 'w_z_frac'

             Plus 'K_vorticity' et 'K_current' SI ET SEULEMENT SI
             `split_plaquette=True`. Par defaut l'ensemble des cles rendues
             est exactement celui d'avant le scindement — voir le parametre
             ci-dessus pour pourquoi c'est le contrat, et pas un detail.
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

        # Normalisation du couplage ZZ.
        #
        # `legacy` divise par la MOYENNE des sauts. `max` divise par le MAX,
        # ce qui borne le terme : max|C| == w_zz exactement, quel que soit
        # `dim`. Voir l'en-tete de la classe pour ce que ce choix change et
        # ce qu'il ne peut pas changer.
        if self.norm == "max":
            # Garde MULTIPLICATIF, pas additif : `+ EPS` decalerait l'echelle
            # et max|C| ne vaudrait plus w_zz qu'a ~1e-10 pres. Ici
            # l'invariance est exacte, et le cas degenere (champ uniforme,
            # aucun saut) rend un champ de zeros plutot qu'un 0/0.
            pic = max(float(np.max(jump_h)), float(np.max(jump_v)))
            norm_jump = pic if pic > self.EPS else 1.0
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

        # Normalisation de la plaquette.
        #
        # `legacy` divise par max|omega| + max|J|, DEUX maxima pris en des
        # points possiblement differents : max|K| < w_zzzz des que les deux
        # extrema ne coincident pas, d'une quantite qui depend du champ.
        # `max` divise par le max de la SOMME, donc max|K| == w_zzzz
        # exactement — c'est ce qui rend le terme comparable d'un `dim` a
        # l'autre.
        signal_plaq = np.abs(omega_z) + np.abs(Jz_curl)
        if self.norm == "max":
            pic_plaq = float(np.max(signal_plaq))
            norm_plaq = pic_plaq if pic_plaq > self.EPS else 1.0
        else:
            norm_plaq = np.max(np.abs(omega_z)) + np.max(np.abs(Jz_curl)) + self.EPS

        K_plaquettes = -self.w_zzzz * signal_plaq / norm_plaq

        # ── Plaquette SCINDEE : un terme par type de structure ────────────
        # `K_plaquettes` somme |omega| et |J|, si bien qu'un vortex pur et
        # une nappe de courant pure rendent EXACTEMENT la meme valeur : le
        # terme capte « il se passe quelque chose », pas un type.
        #
        # Les deux termes ci-dessous separent les deux. Chacun est normalise
        # par le max de SON PROPRE signal — dans les deux modes, y compris
        # `legacy` : les normaliser par une moyenne reintroduirait le
        # couplage entre la spikiness du champ et le poids des familles,
        # qui est precisement ce qu'on retire.
        #
        # `K_plaquettes` est laisse INCHANGE bit a bit. Les deux termes
        # scindes sont OPT-IN (`split_plaquette=True`) : le dict rendu est
        # consomme comme un TOUT par `call_vqa_shell.py` (somme de |coeff|
        # sur toutes les cles -> `E_max`), si bien qu'AJOUTER une cle est
        # deja un changement de comportement, meme quand aucune valeur
        # partagee ne bouge.
        def _selectif(signal):
            pic = float(np.max(np.abs(signal)))
            return -self.w_zzzz * np.abs(signal) / (pic if pic > self.EPS else 1.0)

        split_terms = {}
        if split_plaquette:
            split_terms["K_vorticity"] = _selectif(omega_z)
            split_terms["K_current"] = _selectif(Jz_curl)

        # ==============================================================
        #  3. Z (activity bias)
        #     h_i = +c * median(|C|, |K|) * (s_i - thr)
        #     Signe POSITIF, cf. l'en-tete du module : au-dessus du seuil le
        #     biais pousse vers |1> (raffiner). Cette ligne portait un signe
        #     negatif, contraire au code qui la suit.
        # ==============================================================
        all_coeffs = np.concatenate([
            np.abs(C_horiz).ravel(),
            np.abs(C_vert).ravel(),
            np.abs(K_plaquettes).ravel(),
        ])
        nonzero = all_coeffs[all_coeffs > self.EPS]
        if len(nonzero) == 0:
            echelle = 0.0
        elif self.norm == "max":
            # Accroche le biais au MAX des couplages : le rapport
            # biais/couplage vaut alors exactement `c_bias`, fixe par
            # construction et independant de `dim`. Avec la mediane
            # (`legacy`), ce rapport suit la dissymetrie de la distribution
            # des sauts, qui croit avec la resolution — donc le biais
            # decroche des couplages quand `dim` monte.
            echelle = float(np.max(nonzero))
        else:
            echelle = float(np.median(nonzero))

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
            **split_terms,
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
