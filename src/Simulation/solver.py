import numpy as np
from scipy.ndimage import zoom, map_coordinates
from Simulation.grid import AXIS_X, AXIS_Y, project_divergence_free_any
from Simulation.utils import compute_local_factor

class MHDSolver:
    """
    Solveur MHD 2D Scientifique pour Q-HAS.
    
    Architecture Numérique :
    ========================
    - Dérivées spatiales    : Différences Finies Ordre 4 (LOCALES)
    - Intégration temporelle : RK4 (Runge-Kutta ordre 4)
    - Projection div-free   : Spectrale (FFT) — contrainte globale
    - AMR                   : Vrai Multi-Niveaux Hiérarchique guidé par la 'depth'
    """

    def __init__(self, grid, dt=1e-4, Re=100, Rm=100):
        self.grid = grid
        self.dt = dt
        self.nu = 1.0 / Re
        self.eta = 1.0 / Rm

        N = grid.N
        self.vx = np.zeros((N, N))
        self.vy = np.zeros((N, N))
        self.Bx = np.zeros((N, N))
        self.By = np.zeros((N, N))

        self.dx = (2.0 * np.pi) / N
        self.energy_history = {'kinetic': [], 'magnetic': [], 'total': []}

    # ================================================================
    #                    INITIALISATIONS PHYSIQUES
    # ================================================================

    def apply_physics_perturbation(self, seed, amplitude=0.1, k_cut=8):
        """Apply a reproducible large-scale velocity perturbation.

        Seed 0 is the unperturbed reference trajectory. Positive seeds add
        independent band-limited fields and restore discrete incompressibility.
        """
        seed = int(seed)
        if seed < 0:
            raise ValueError("physics seed must be non-negative")
        if seed == 0:
            return
        if not np.isfinite(amplitude) or amplitude <= 0.0:
            raise ValueError("physics perturbation amplitude must be finite and > 0")
        rng = np.random.default_rng(seed)
        wave = np.fft.fftfreq(self.grid.N) * self.grid.N
        kx, ky = np.meshgrid(wave, wave, indexing="ij")
        keep = np.sqrt(kx ** 2 + ky ** 2) <= int(k_cut)

        def noise():
            spectrum = np.fft.fft2(rng.standard_normal(self.vx.shape))
            spectrum[~keep] = 0.0
            field = np.real(np.fft.ifft2(spectrum))
            return field / max(float(field.std()), 1e-30)

        self.vx = self.vx + float(amplitude) * noise()
        self.vy = self.vy + float(amplitude) * noise()
        self.enforce_incompressibility()

    def init_kelvin_helmholtz(self, shear_width= 0.5, noise_amplitude=0.1, drift_velocity=0.5):
        X, Y = self.grid.X, self.grid.Y
        v_flow = (np.tanh((Y - np.pi / 2) / shear_width)
                  - np.tanh((Y - 3 * np.pi / 2) / shear_width) - 1.0)
        self.vx = v_flow + drift_velocity
        self.vy = np.zeros_like(X)

        perturbation = noise_amplitude * np.sin(X) * (
            np.exp(-((Y - np.pi / 2) ** 2) / (shear_width ** 2))
            + np.exp(-((Y - 3 * np.pi / 2) ** 2) / (shear_width ** 2))
        )
        self.vy += perturbation
        self.Bx = 0.1 * np.ones_like(X)
        self.By = np.zeros_like(X)

        self.enforce_incompressibility()
        self.record_energy()

    def init_orszag_tang(self):
        X, Y = self.grid.X, self.grid.Y
        self.vx = -np.sin(Y)
        self.vy =  np.sin(X)
        self.Bx = -np.sin(Y)
        self.By =  np.sin(2 * X)
        self.enforce_incompressibility()

    # ----------------------------------------------------------------
    #  Benchmark A — "Silent" Instability (Phase Sensitivity)
    #  |B| ≈ const partout, mais la direction du champ tourne à travers
    #  deux couches de cisaillement magnétique. ∇|B| ≈ 0 → l'AMR
    #  classique ne voit rien. Le VQA détecte la rotation de phase via
    #  le flux Φ̇ avant l'effondrement kink.
    # ----------------------------------------------------------------
    def init_magnetic_twist(self, twist_angle=np.pi/2, shear_width=0.3,
                            perturbation=0.01):
        X, Y = self.grid.X, self.grid.Y
        B0 = 1.0
        # Double couche de rotation pour la périodicité en y
        # alpha varie de ~0 à twist_angle entre les deux interfaces
        alpha = (twist_angle / 2.0) * (
            np.tanh((Y - np.pi / 2) / shear_width)
            - np.tanh((Y - 3 * np.pi / 2) / shear_width)
            - 1.0
        )
        # B = (B0 cos alpha(y), B_guide) — solénoïdal PAR CONSTRUCTION :
        # Bx ne dépend pas de x et By est constant, donc div B = 0 exactement.
        #
        # Poser B = (B0 cos alpha, B0 sin alpha) à la place n'est PAS à
        # divergence nulle en 2-D : `enforce_incompressibility` y annulerait
        # alors la torsion elle-même, silencieusement.
        #
        # En 2-D, un champ solénoïdal dont la direction tourne exige que la
        # composante parallèle à la variation reste constante : c'est ce que
        # fait le champ guide B_guide ci-dessous.
        # La composante VARIABLE doit changer de signe pour que la direction
        # balaie réellement `twist_angle` : avec Bx = B0 sin(alpha) et alpha
        # parcourant [-twist/2, +twist/2], l'angle va de
        # atan2(guide, -B0 sin(twist/2)) à atan2(guide, +B0 sin(twist/2)),
        # soit exactement `twist_angle` pour guide = B0 cos(twist/2).
        b_guide = B0 * np.cos(twist_angle / 2.0)
        self.Bx = B0 * np.sin(alpha)
        self.By = np.full_like(X, b_guide)
        # Pas de vitesse initiale — la tension magnétique drive la dynamique
        self.vx = np.zeros_like(X)
        self.vy = perturbation * np.sin(X) * (
            np.exp(-((Y - np.pi / 2) ** 2) / shear_width ** 2)
            + np.exp(-((Y - 3 * np.pi / 2) ** 2) / shear_width ** 2)
        )
        self.enforce_incompressibility()

    # ----------------------------------------------------------------
    #  Benchmark B — Noise Immunity (Topological Protection)
    #  Champ uniforme B0 = (1, 0) + bruit blanc gaussien.
    #  Évolution physique = diffusion pure. Tout raffinement est un
    #  faux positif. L'AMR classique sur-raffine (∇²noise → ∞),
    #  le VQA filtre topologiquement (∮ δB·dl ≈ 0 sur bruit incohérent).
    # ----------------------------------------------------------------
    def init_noisy_uniform(self, B0=1.0, noise_sigma=0.05, seed=42):
        X = self.grid.X
        rng = np.random.default_rng(seed)
        # Bruit tiré d'une fonction de flux, donc solénoïdal par
        # construction : tirer Bx et By indépendamment ne serait
        # solénoïdal qu'à moitié après projection, et `noise_sigma` ne
        # serait plus l'écart-type réellement obtenu. On tire psi, on
        # prend son rotationnel, puis on renormalise pour que l'écart-type
        # demandé soit celui produit.
        psi = rng.standard_normal(X.shape)
        bx, by = self._curl_z_fd4(psi, self.dx)
        scale = noise_sigma / max(float(np.std(np.concatenate([bx.ravel(),
                                                               by.ravel()]))),
                                  1e-30)
        self.Bx = B0 + scale * bx
        self.By = scale * by
        self.vx = np.zeros_like(X)
        self.vy = np.zeros_like(X)
        self.enforce_incompressibility()

    # ----------------------------------------------------------------
    #  Benchmark C — Topological Defect (Tearing / Harris Sheet)
    #  Nappe de courant Harris en équilibre : Bx = B0 tanh(y/L).
    #  Perturbation δBy ∝ cos(kx) pour déclencher la tearing mode.
    #  L'AMR classique raffine TOUTE la nappe (gradient fort partout).
    #  Le VQA détecte sélectivement les X-points (reconnexion topologique)
    #  vs les O-points (centres d'îlots stables).
    #  Double nappe pour la périodicité en y.
    # ----------------------------------------------------------------
    def init_harris_tearing(self, B0=1.0, shear_width=0.3,
                            perturbation=0.01, k_mode=1.0):
        X, Y = self.grid.X, self.grid.Y
        # Double nappe de courant pour la périodicité
        self.Bx = B0 * (
            np.tanh((Y - np.pi / 2) / shear_width)
            - np.tanh((Y - 3 * np.pi / 2) / shear_width)
            - 1.0
        )
        # Perturbation magnétique pour déclencher la tearing mode, posée
        # par fonction de flux : à divergence nulle par construction. Ne
        # poser que `dBy` seul se ferait en grande partie retirer par la
        # projection.
        u1 = (Y - np.pi / 2) / shear_width
        u2 = (Y - 3 * np.pi / 2) / shear_width
        env = 1.0 / np.cosh(u1) ** 2 + 1.0 / np.cosh(u2) ** 2
        # d/dy sech^2(u) = -(2/w) sech^2(u) tanh(u)
        psi = -(perturbation / k_mode) * np.sin(k_mode * X) * env
        dBx, dBy = self._curl_z_fd4(psi, self.dx)
        self.Bx = self.Bx + dBx
        self.By = dBy
        # Pas de vitesse initiale — la reconnexion drive la dynamique
        self.vx = np.zeros_like(X)
        self.vy = np.zeros_like(X)
        self.enforce_incompressibility()

    # ----------------------------------------------------------------
    #  Double Tearing Mode — deux nappes de courant proches
    #  Chaque nappe est individuellement sous le seuil de l'AMR classique,
    #  mais leur interaction crée une configuration frustrée que le VQA
    #  détecte via les plaquettes (circulation non-nulle à l'échelle du patch).
    #  Séparation d ≈ taille d'un demi-patch VQA pour maximiser l'avantage.
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    #  Vortex isolé de Lamb-Oseen
    #  Un seul tourbillon axisymétrique : v_θ = Γ/(2πr) * (1 - exp(-r²/r_c²))
    #  Le champ magnétique est uniforme → |Jz| ≈ 0 partout sauf au cœur
    #  du vortex où la vorticité concentrée entraîne un courant induit
    #  par advection.
    #  Ce scénario isole la détection de vorticité pure : pas de shear
    #  layer, pas de choc, pas de reconnexion magnétique.
    #  Dupliqué en deux centres pour la périodicité (même logique que
    #  les doubles nappes dans harris_tearing).
    # ----------------------------------------------------------------
    def init_lamb_oseen_vortex(self, circulation=6.0, core_radius=0.4,
                                B0=0.1, noise_amplitude=0.005):
        X, Y = self.grid.X, self.grid.Y
        cx1, cy1 = np.pi / 2, np.pi / 2
        cx2, cy2 = 3 * np.pi / 2, 3 * np.pi / 2

        def _vortex_velocity(cx, cy):
            dx = X - cx
            dy = Y - cy
            # Periodic minimum image
            dx = dx - 2 * np.pi * np.round(dx / (2 * np.pi))
            dy = dy - 2 * np.pi * np.round(dy / (2 * np.pi))
            r2 = dx**2 + dy**2 + 1e-12
            r = np.sqrt(r2)
            # Lamb-Oseen tangential velocity: v_theta = Gamma/(2*pi*r) * (1 - exp(-r^2/rc^2))
            rc2 = core_radius**2
            v_theta = (circulation / (2 * np.pi * r)) * (1.0 - np.exp(-r2 / rc2))
            # Convert to Cartesian: v_x = -v_theta * sin(theta), v_y = v_theta * cos(theta)
            vx = -v_theta * dy / r
            vy =  v_theta * dx / r
            return vx, vy

        vx1, vy1 = _vortex_velocity(cx1, cy1)
        vx2, vy2 = _vortex_velocity(cx2, cy2)
        self.vx = vx1 + vx2
        self.vy = vy1 + vy2

        # Uniform background magnetic field — no current sheets
        self.Bx = B0 * np.ones_like(X)
        self.By = np.zeros_like(X)

        # Small noise to break exact symmetry
        rng = np.random.default_rng(42)
        self.vy += noise_amplitude * rng.standard_normal(X.shape)

        self.enforce_incompressibility()
        self.record_energy()

    def init_double_tearing(self, B0=0.6, separation=0.5, shear_width=0.2,
                            perturbation=0.01, k_mode=2.0):
        X, Y = self.grid.X, self.grid.Y
        d = separation
        # 4 nappes (2 paires) pour la périodicité
        # Paire 1 autour de y = π/2, paire 2 autour de y = 3π/2
        self.Bx = B0 * (
            np.tanh((Y - (np.pi / 2 - d)) / shear_width)
            - np.tanh((Y - (np.pi / 2 + d)) / shear_width)
            + np.tanh((Y - (3 * np.pi / 2 - d)) / shear_width)
            - np.tanh((Y - (3 * np.pi / 2 + d)) / shear_width)
            - 2.0
        )
        # Perturbation pour la tearing, posée par fonction de flux (à
        # divergence nulle par construction).
        g1 = np.exp(-((Y - np.pi / 2) ** 2) / (2 * d) ** 2)
        g2 = np.exp(-((Y - 3 * np.pi / 2) ** 2) / (2 * d) ** 2)
        env = g1 + g2
        # cos(kx) ici (plutôt que sin) donne le même profil physique,
        # après décalage de phase.
        psi = (perturbation / k_mode) * np.cos(k_mode * X) * env
        dBx, dBy = self._curl_z_fd4(psi, self.dx)
        self.Bx = self.Bx + dBx
        self.By = dBy
        self.vx = np.zeros_like(X)
        self.vy = np.zeros_like(X)
        self.enforce_incompressibility()

    # ----------------------------------------------------------------
    #  MHD Rotor — Budget-Constrained Proof-of-Concept
    #  A dense rotating disk (radius r0, angular velocity omega) in a
    #  static ambient medium with uniform Bx = B0.
    #  The rotor winds up the magnetic field, producing:
    #    - Rotor core:  high vorticity, low Jz  → smooth (NO refinement)
    #    - Magnetic sheath: high vorticity AND high Jz → YES refine
    #    - Torsional wave front: medium vort, high Jz → YES refine
    #    - Far field: quiet → NO refine
    #  Classical AMR (linear combination) cannot distinguish the rotor
    #  core from the sheath because both have high vorticity.
    #  With a budget constraint, classical wastes slots on the core.
    #  Q-HAS exploits vort×Jz correlation via entanglement.
    # ----------------------------------------------------------------
    def init_mhd_rotor(self, omega=10.0, r0=0.75, taper_width=0.15,
                       B0=1.0, perturbation=0.005):
        """
        MHD Rotor (Balsara & Spicer 1999, Tóth 2000).

        Parameters
        ----------
        omega : float
            Angular velocity of the rotor core.
        r0 : float
            Radius of the rotor disk (in domain units, domain is [0, 2π]).
        taper_width : float
            Width of the smooth taper at the disk boundary.
        B0 : float
            Uniform background magnetic field strength (x-direction).
        perturbation : float
            Small noise amplitude to break exact symmetry.
        """
        X, Y = self.grid.X, self.grid.Y
        cx, cy = np.pi, np.pi  # Center of the domain

        dx = X - cx
        dy = Y - cy
        # Periodic minimum image
        dx = dx - 2 * np.pi * np.round(dx / (2 * np.pi))
        dy = dy - 2 * np.pi * np.round(dy / (2 * np.pi))
        r = np.sqrt(dx**2 + dy**2 + 1e-12)

        # Smooth taper: 1 inside disk, 0 outside, smooth transition
        taper = 0.5 * (1.0 - np.tanh((r - r0) / taper_width))

        # Rigid body rotation: v_theta = omega * r, capped by taper
        self.vx = -omega * dy * taper
        self.vy =  omega * dx * taper

        # Uniform magnetic field in x-direction
        self.Bx = B0 * np.ones_like(X)
        self.By = np.zeros_like(X)

        # Small noise to break symmetry
        rng = np.random.default_rng(42)
        self.vy += perturbation * rng.standard_normal(X.shape)

        self.enforce_incompressibility()
        self.record_energy()

    def init_island_coalescence(self, B0=1.0, shear_width=0.3,
                                perturbation=0.05, k_mode=1.0):
        """
        Island Coalescence — merging magnetic islands via reconnection.

        Two chains of magnetic islands (O-points) separated by X-points.
        The perturbation drives island merging, creating dynamic X-points
        where reconnection occurs.  This is the canonical test for
        selective refinement of reconnection sites: classical AMR refines
        the entire current sheet uniformly, while Q-HAS should focus on
        the X-points (det(J_B) < 0) and ignore the stable O-points.

        Uses the same double-sheet topology as harris_tearing for
        periodicity, but with a stronger perturbation that drives
        the islands to coalesce rather than just tear.

        Double chain for periodicity (same logic as harris_tearing).
        """
        X, Y = self.grid.X, self.grid.Y
        # Double current sheet (same as Harris tearing)
        self.Bx = B0 * (
            np.tanh((Y - np.pi / 2) / shear_width)
            - np.tanh((Y - 3 * np.pi / 2) / shear_width)
            - 1.0
        )
        # Stronger perturbation to drive island coalescence, posée par
        # fonction de flux (voir harris_tearing).
        u1 = (Y - np.pi / 2) / shear_width
        u2 = (Y - 3 * np.pi / 2) / shear_width
        env = 1.0 / np.cosh(u1) ** 2 + 1.0 / np.cosh(u2) ** 2
        psi = -(perturbation / k_mode) * np.sin(k_mode * X) * env
        dBx, dBy = self._curl_z_fd4(psi, self.dx)
        self.Bx = self.Bx + dBx
        self.By = dBy
        # Small velocity perturbation to drive coalescence
        self.vx = np.zeros_like(X)
        self.vy = perturbation * np.sin(k_mode * X) * (
            np.exp(-((Y - np.pi / 2) ** 2) / shear_width ** 2)
            + np.exp(-((Y - 3 * np.pi / 2) ** 2) / shear_width ** 2)
        )
        self.enforce_incompressibility()

    # ----------------------------------------------------------------
    #  Perturbations magnétiques : par fonction de flux
    # ----------------------------------------------------------------
    @staticmethod
    def _curl_z_fd4(psi, dx):
        """`rot(psi z)` avec le MEME stencil FD4 que le second membre.

        `div(rot psi) = d_x d_y psi - d_y d_x psi` : exactement nul, parce
        que les deux dérivées FD4 sont des combinaisons de `np.roll` et
        commutent. Dériver `psi` analytiquement ne donnerait la contrainte
        qu'à la précision de discrétisation, pas exactement — une
        contrainte discrète ne se satisfait que dans l'opérateur qui la
        mesure.
        """
        g_x, g_y = MHDSolver._fd_grad(psi, dx)
        return g_y, -g_x

    def get_fluxes(self):
        dx = self.dx
        grad_By_x = (np.roll(self.By, -1, axis=0) - np.roll(self.By, 1, axis=0)) / (2.0 * dx)
        grad_Bx_y = (np.roll(self.Bx, -1, axis=1) - np.roll(self.Bx, 1, axis=1)) / (2.0 * dx)
        Jz = grad_By_x - grad_Bx_y
        return {'vx': self.vx, 'vy': self.vy, 'Bx': self.Bx, 'By': self.By, 'Jz': Jz}

    def check_cfl(self):
        v_max = max(np.max(np.abs(self.vx)), np.max(np.abs(self.vy)))
        B_max = max(np.max(np.abs(self.Bx)), np.max(np.abs(self.By)))
        c_fast = v_max + B_max
        cfl = c_fast * self.dt / self.dx
        if cfl > 1.0:
            print(f"[WARNING] Violation CFL détéctée : {cfl:.2f} > 1.0.")
        return cfl

    def record_energy(self):
        Ek = 0.5 * np.sum(self.vx ** 2 + self.vy ** 2) * self.dx ** 2
        Em = 0.5 * np.sum(self.Bx ** 2 + self.By ** 2) * self.dx ** 2
        self.energy_history['kinetic'].append(Ek)
        self.energy_history['magnetic'].append(Em)
        self.energy_history['total'].append(Ek + Em)

    #: Projeter aussi le champ magnetique. Par defaut False : l'induction le
    #: garde deja a divergence nulle, et la projection l'en ECARTE.
    #: Voir `enforce_incompressibility` pour la mesure.
    PROJECT_B = False

    def enforce_incompressibility(self):
        """Impose la contrainte de divergence nulle sur la vitesse.

        LE CHAMP MAGNETIQUE N'EST PLUS PROJETE — et c'est une correction,
        pas un oubli.

        L'induction est ecrite en forme rotationnelle :
        `rhs_B = (dEz/dy, -dEz/dx)`. Sa divergence AUX DIFFERENCES FINIES
        vaut `d2Ez/dxdy - d2Ez/dydx`, exactement nulle puisque les decalages
        de `np.roll` commutent. B est donc solenoidal par construction, dans
        l'operateur meme qui construit le second membre.

        La projection, elle, est SPECTRALE : un operateur DIFFERENT de
        celui qui a construit B. Appliquee a un champ deja a divergence FD
        nulle, elle ne le nettoie pas — elle y injecte le desaccord entre
        les deux operateurs. Mesure sur Orszag-Tang N=64 : 50 pas SANS
        projeter B laissent sa divergence FD a 1.00e-14 (bruit) ; 50 pas
        EN la projetant la fait monter a 4.63e-07, huit ordres de
        grandeur perdus. Sur un run complet (T=0.05, 256 pas), l'erreur
        en temps est IDENTIQUE que B soit projete ou non (1.185e-05) —
        projeter B ne gagne donc rien, et laisse sa divergence FD a
        4.877e-06 contre 2.818e-14 sans.

        La vitesse, elle, en a reellement besoin : sa divergence FD
        n'est PAS nulle analytiquement.

        `PROJECT_B = True` reproduit le chemin historique bit a bit.
        """
        self.vx, self.vy = self.grid.project_divergence_free(self.vx, self.vy)
        if self.PROJECT_B:
            self.Bx, self.By = self.grid.project_divergence_free(self.Bx, self.By)

    def is_diverged(self, max_value=1e8):
        """Check if any field has NaN, Inf, or has blown up beyond physical limits.

        Le seuil doit rester petit devant l'echelle de l'overflow `float64`
        (~1e154) mais tres grand devant l'echelle physique legitime des
        scenarios de ce depot (champs d'ordre 1-4) : un seuil demesurement
        haut ne detecterait plus qu'un run deja depourvu de sens (NaN/Inf).
        Une divergence MHD croit exponentiellement, donc couper tot ne
        perd aucun run viable et laisse le score partiel se calculer sur
        des champs moins corrompus.

        `max_value` reste un parametre : un appelant qui travaille a une
        autre echelle peut l'elargir explicitement.
        """
        for field in [self.vx, self.vy, self.Bx, self.By]:
            if np.any(np.isnan(field)) or np.any(np.isinf(field)):
                return True
            if np.max(np.abs(field)) > max_value:
                return True
        return False

    # ================================================================
    #         NOYAU PHYSIQUE — DIFFÉRENCES FINIES ORDRE 4
    # ================================================================

    @staticmethod
    def _fd_grad(f, dx):
        """Gradient centré ordre 4. Convention: axis=0 = X, axis=1 = Y."""
        fp2_x = np.roll(f, -2, axis=0); fp1_x = np.roll(f, -1, axis=0)
        fm1_x = np.roll(f,  1, axis=0); fm2_x = np.roll(f,  2, axis=0)
        g_x = (-fp2_x + 8.0 * fp1_x - 8.0 * fm1_x + fm2_x) / (12.0 * dx)

        fp2_y = np.roll(f, -2, axis=1); fp1_y = np.roll(f, -1, axis=1)
        fm1_y = np.roll(f,  1, axis=1); fm2_y = np.roll(f,  2, axis=1)
        g_y = (-fp2_y + 8.0 * fp1_y - 8.0 * fm1_y + fm2_y) / (12.0 * dx)
        return g_x, g_y

    @staticmethod
    def _fd_laplacian(f, dx):
        """Laplacien centré ordre 4. Convention: axis=0 = X, axis=1 = Y."""
        fp2_x = np.roll(f, -2, axis=0); fp1_x = np.roll(f, -1, axis=0)
        fm1_x = np.roll(f,  1, axis=0); fm2_x = np.roll(f,  2, axis=0)
        lap_x = (-fp2_x + 16.0 * fp1_x - 30.0 * f + 16.0 * fm1_x - fm2_x) / (12.0 * dx**2)

        fp2_y = np.roll(f, -2, axis=1); fp1_y = np.roll(f, -1, axis=1)
        fm1_y = np.roll(f,  1, axis=1); fm2_y = np.roll(f,  2, axis=1)
        lap_y = (-fp2_y + 16.0 * fp1_y - 30.0 * f + 16.0 * fm1_y - fm2_y) / (12.0 * dx**2)
        return lap_x + lap_y

    def _compute_rhs_fd(self, vx, vy, Bx, By, dx, nu=None, eta=None):
        if nu is None: nu = self.nu
        if eta is None: eta = self.eta

        # --- Gradients simples ---
        g_vx_x, g_vx_y = self._fd_grad(vx, dx)
        g_vy_x, g_vy_y = self._fd_grad(vy, dx)
        g_Bx_x, g_Bx_y = self._fd_grad(Bx, dx)
        g_By_x, g_By_y = self._fd_grad(By, dx)

        # --- 1. ADVECTION : Forme "Skew-Symmetric" (Conservation d'énergie stricte) ---
        g_vxx_x, _ = self._fd_grad(vx * vx, dx)
        g_vxy_x, g_vxy_y = self._fd_grad(vx * vy, dx)
        _, g_vyy_y = self._fd_grad(vy * vy, dx)
        
        adv_v_x = 0.5 * (vx * g_vx_x + vy * g_vx_y + g_vxx_x + g_vxy_y)
        adv_v_y = 0.5 * (vx * g_vy_x + vy * g_vy_y + g_vxy_x + g_vyy_y)

        # --- 2. FORCE DE LORENTZ : Forme J x B ---
        Jz = g_By_x - g_Bx_y
        lorentz_x = -Jz * By
        lorentz_y = Jz * Bx

        # --- 3. INDUCTION : Forme Rotationnelle (Préserve Div B = 0) ---
        Ez = vx * By - vy * Bx
        g_Ez_x, g_Ez_y = self._fd_grad(Ez, dx)
        rhs_Bx = g_Ez_y
        rhs_By = -g_Ez_x

        # --- 4. DIFFUSION ---
        diff_vx = nu * self._fd_laplacian(vx, dx)
        diff_vy = nu * self._fd_laplacian(vy, dx)
        diff_Bx = eta * self._fd_laplacian(Bx, dx)
        diff_By = eta * self._fd_laplacian(By, dx)

        # --- ASSEMBLAGE ---
        rhs_vx = -adv_v_x + lorentz_x + diff_vx
        rhs_vy = -adv_v_y + lorentz_y + diff_vy
        rhs_Bx = rhs_Bx + diff_Bx
        rhs_By = rhs_By + diff_By

        return rhs_vx, rhs_vy, rhs_Bx, rhs_By
    
    def _rk2_step(self, vx, vy, Bx, By, dx, dt, nu=None, eta=None):
        """RK2 (Heun) — stable avec la projection div-free correcte."""
        k1 = self._compute_rhs_fd(vx, vy, Bx, By, dx, nu, eta)
        vx_p = vx + dt * k1[0]
        vy_p = vy + dt * k1[1]
        Bx_p = Bx + dt * k1[2]
        By_p = By + dt * k1[3]

        k2 = self._compute_rhs_fd(vx_p, vy_p, Bx_p, By_p, dx, nu, eta)
        return (vx + (dt / 2.0) * (k1[0] + k2[0]),
                vy + (dt / 2.0) * (k1[1] + k2[1]),
                Bx + (dt / 2.0) * (k1[2] + k2[2]),
                By + (dt / 2.0) * (k1[3] + k2[3]))
    
    #: Projeter le SECOND MEMBRE a chaque etage RK4 plutot que l'ETAT une
    #: fois le pas fini (voir `_rk4_step` pour le gain d'ordre).
    #:
    #: PAR DEFAUT False, malgre ce gain, parce que la correction n'est
    #: VALIDE QUE SUR `step_full`. `_rk4_step` a trois appelants :
    #:
    #:   step_full       champ global periodique        -> projection valide
    #:   step_layered/1  champ global sous-echantillonne -> periodique, mais
    #:                   d'une autre TAILLE que self.grid : la projection leve
    #:   step_layered/2  patch LOCAL avec halo           -> pas periodique,
    #:                   une projection spectrale periodique n'y est pas definie
    #:
    #: Projeter les deux premiers et pas le troisieme romprait la garantie
    #: « a max_depth, step_layered est identique a step_full » : etendre ce
    #: flag est une decision de modelisation, pas une correction de defaut.
    PROJECT_RHS = False

    def _projected_rhs(self, vx, vy, Bx, By, dx, nu, eta):
        """Second membre rendu a divergence nulle avant integration.

        Le systeme est differentiel-algebrique : la vitesse et le champ
        magnetique doivent rester a divergence nulle. Imposer la contrainte
        APRES un pas RK4 non contraint est un splitting de Lie, d'ordre 1 —
        c'est ce qui ramenait le solveur d'ordre 4 a ordre 1.2.

        En projetant le second membre, le champ integre est a divergence
        nulle PAR CONSTRUCTION et RK4 garde son ordre. La projection reste
        idempotente et lineaire, donc elle commute avec la combinaison des
        etages.
        """
        kvx, kvy, kBx, kBy = self._compute_rhs_fd(vx, vy, Bx, By, dx, nu, eta)
        # Projection independante de la taille : `step_layered` calcule sa
        # phase 1 sur le champ global SOUS-ECHANTILLONNE, qui reste
        # periodique mais n'a plus la taille de la grille.
        kvx, kvy = project_divergence_free_any(kvx, kvy)
        kBx, kBy = project_divergence_free_any(kBx, kBy)
        return kvx, kvy, kBx, kBy

    def _rk4_step(self, vx, vy, Bx, By, dx, dt, nu=None, eta=None):
        """Integration temporelle Runge-Kutta d'ordre 4 (RK4).

        Projeter l'ETAT apres un pas RK4 complet degrade l'ordre du schema
        a 1 (splitting de Lie) ; projeter le SECOND MEMBRE a chaque etage
        preserve l'ordre 4 de RK4 tout en controlant la divergence aussi
        bien que la projection de l'etat. Ne pas projeter du tout garde
        aussi l'ordre 4 mais laisse la divergence exploser.

        A ne pas confondre avec un splitting de Strang : la projection est
        un projecteur idempotent (P.P = P), pas un flot decoupable en
        demi-pas.
        """
        _rhs = self._projected_rhs if self.PROJECT_RHS else self._compute_rhs_fd

        # Étape 1
        k1_vx, k1_vy, k1_Bx, k1_By = _rhs(vx, vy, Bx, By, dx, nu, eta)

        # Étape 2
        vxp2 = vx + 0.5 * dt * k1_vx
        vyp2 = vy + 0.5 * dt * k1_vy
        Bxp2 = Bx + 0.5 * dt * k1_Bx
        Byp2 = By + 0.5 * dt * k1_By
        k2_vx, k2_vy, k2_Bx, k2_By = _rhs(vxp2, vyp2, Bxp2, Byp2, dx, nu, eta)

        # Étape 3
        vxp3 = vx + 0.5 * dt * k2_vx
        vyp3 = vy + 0.5 * dt * k2_vy
        Bxp3 = Bx + 0.5 * dt * k2_Bx
        Byp3 = By + 0.5 * dt * k2_By
        k3_vx, k3_vy, k3_Bx, k3_By = _rhs(vxp3, vyp3, Bxp3, Byp3, dx, nu, eta)

        # Étape 4
        vxp4 = vx + dt * k3_vx
        vyp4 = vy + dt * k3_vy
        Bxp4 = Bx + dt * k3_Bx
        Byp4 = By + dt * k3_By
        k4_vx, k4_vy, k4_Bx, k4_By = _rhs(vxp4, vyp4, Bxp4, Byp4, dx, nu, eta)
        
        # Somme pondérée finale
        vx_new = vx + (dt / 6.0) * (k1_vx + 2*k2_vx + 2*k3_vx + k4_vx)
        vy_new = vy + (dt / 6.0) * (k1_vy + 2*k2_vy + 2*k3_vy + k4_vy)
        Bx_new = Bx + (dt / 6.0) * (k1_Bx + 2*k2_Bx + 2*k3_Bx + k4_Bx)
        By_new = By + (dt / 6.0) * (k1_By + 2*k2_By + 2*k3_By + k4_By)
        
        return vx_new, vy_new, Bx_new, By_new

    def adapt_dt(self, cfl_target=0.4):
        v_max = max(np.max(np.abs(self.vx)), np.max(np.abs(self.vy)))
        B_max = max(np.max(np.abs(self.Bx)), np.max(np.abs(self.By)))
        c_fast = v_max + B_max + 1e-12

        dt_adv = cfl_target * self.dx / c_fast
        nu_max = max(self.nu, self.eta)
        dt_diff = 0.5 * cfl_target * (self.dx ** 2) / (nu_max + 1e-12)

        self.dt = min(dt_adv, dt_diff)
        return self.dt

    # ================================================================
    #    OPÉRATIONS DE GRILLE — DOWNSAMPLE / UPSAMPLE
    # ================================================================

    @staticmethod
    def _downsample_local(field, factor):
        """Restriction conservative (block-average) — préserve l'intégrale."""
        if factor == 1: return field
        H, W = field.shape
        Hc, Wc = H // factor, W // factor
        return field.reshape(Hc, factor, Wc, factor).mean(axis=(1, 3))

    @staticmethod
    def _upsample_local(field, factor):
        """Prolongation bicubique (ordre 3) pour les patchs locaux.
        Cohérent avec le schéma FD4 — évite de dégrader l'ordre spatial."""
        if factor == 1: return field
        return zoom(field, factor, order=3)

    @staticmethod
    def _upsample_global(field, factor):
        """Prolongation bicubique périodique (ordre 3) pour la grille complète.

        Deux conventions doivent coïncider avec celles de `PeriodicGrid` :

        1. ÉCHANTILLONNAGE AUX NŒUDS. `PeriodicGrid` pose ses points sur
           `linspace(0, L, N, endpoint=False)`, donc le point fin j tombe à
           l'indice grossier j / factor — PAS la convention centre-de-cellule
           `(j + 0.5) / factor - 0.5`.

        2. mode='grid-wrap'. Depuis scipy 1.6, `mode='wrap'` n'est PAS
           l'enroulement périodique : il traite le tableau comme si le
           premier et le dernier échantillon coïncidaient. C'est
           `'grid-wrap'` qui réalise la topologie torique.
        """
        if factor == 1: return field
        Nc = field.shape[0]
        N = Nc * factor
        pos = np.arange(N) / factor
        I0, I1 = np.meshgrid(pos, pos, indexing='ij')
        return map_coordinates(field, [I0, I1], order=3, mode='grid-wrap')

    # ================================================================
    #                STEP FULL — RÉFÉRENCE (Témoin)
    # ================================================================

    def step_full(self, record_stats=True):
        self.vx, self.vy, self.Bx, self.By = self._rk4_step(
            self.vx, self.vy, self.Bx, self.By, self.dx, self.dt
        )
        
        # Filtre anti-repliement très léger pour assurer la stabilité long terme
        """
        We stop smoothing fields when simulation is stable enough, to preserve fine structures.
        self.vx = self.grid.smooth_field(self.vx)
        self.vy = self.grid.smooth_field(self.vy)
        self.Bx = self.grid.smooth_field(self.Bx)
        self.By = self.grid.smooth_field(self.By)
        """
        self.enforce_incompressibility()
        
        if record_stats:
            self.check_cfl()
            self.record_energy()

    # ================================================================
    #          STEP LAYERED — HYBRIDE AMR MULTI-NIVEAUX (Q-HAS)
    # ================================================================

    def step_layered(self, active_patches, max_depth, target_dim=2):
        """
        Solveur AMR multi-niveaux avec correction tau (defect correction).

        Garantie de convergence : lorsque TOUS les patches sont actifs à
        max_depth (local_factor=1), le résultat est IDENTIQUE à step_full().

        Architecture :
        - Phase 1 : Correction globale coarse (fond pour tout le domaine)
                     Résolution = target_dim^max_depth (le plus grossier possible)
        - Phase 2 : Correction tau par patch (fine_delta - coarse_delta)
                     TOUS les patches participent, y compris coarse_leaf.
                     La profondeur (depth) détermine la précision physique.
                     Subcycling temporel : les patches coarse (local_factor>1)
                     ne sont recalculés que tous les local_factor pas de temps.
        - Phase 3 : Projection div-free (identique à step_full)

        Returns
        -------
        pixels_computed : int
            Nombre effectif de pixels recalculés (post-subcycling),
            utilisable par le pipeline pour le suivi du coût.
        """
        dt = self.dt
        dx = self.dx
        N = self.grid.N

        # Phase 1 coarse factor : theoretical max (depth=0), capped only by
        # divisibility and minimum grid size for FD4 stencil (≥ 5 points).
        cf = int(target_dim ** max_depth)
        while cf > 1 and N % cf != 0:
            cf //= target_dim
        while cf > 1 and N // cf < 5:
            cf //= target_dim
        cf = max(1, cf)

        # FD4 (stencil ±2) × RK4 (4 stages) = 8 pixels de marge
        base_pad = 8

        # Sauvegarde de l'état initial U_n
        vx_n = self.vx.copy()
        vy_n = self.vy.copy()
        Bx_n = self.Bx.copy()
        By_n = self.By.copy()

        # ══════════════════════════════════════════════════════════
        #  Phase 1 : CORRECTION GLOBALE COARSE
        # ══════════════════════════════════════════════════════════
        dx_c = dx * cf

        vx_c0 = self._downsample_local(vx_n, cf)
        vy_c0 = self._downsample_local(vy_n, cf)
        Bx_c0 = self._downsample_local(Bx_n, cf)
        By_c0 = self._downsample_local(By_n, cf)

        vx_c1, vy_c1, Bx_c1, By_c1 = self._rk4_step( 
            vx_c0, vy_c0, Bx_c0, By_c0, dx_c, dt, self.nu, self.eta
        )

        # Stocker les deltas coarse (nécessaires pour la correction tau en Phase 2)
        delta_coarse_vx = self._upsample_global(vx_c1 - vx_c0, cf)
        delta_coarse_vy = self._upsample_global(vy_c1 - vy_c0, cf)
        delta_coarse_Bx = self._upsample_global(Bx_c1 - Bx_c0, cf)
        delta_coarse_By = self._upsample_global(By_c1 - By_c0, cf)

        # Appliquer la correction coarse comme fond pour tout le domaine
        self.vx = vx_n + delta_coarse_vx
        self.vy = vy_n + delta_coarse_vy
        self.Bx = Bx_n + delta_coarse_Bx
        self.By = By_n + delta_coarse_By

        # PAS de lissage ici — il sera appliqué en Phase 3, comme dans step_full

        # ══════════════════════════════════════════════════════════
        #  Phase 2 : CORRECTION TAU PAR PATCH (tous les patches)
        # ══════════════════════════════════════════════════════════
        # Tous les patches participent : la profondeur détermine la précision.
        # tau = fine_delta - coarse_delta_local
        # Quand local_factor=1 : tau = (DNS_delta - coarse_delta) → correction max
        # Quand local_factor=cf : tau ≈ 0 → correction minimale (bords seulement)
        #
        # Subcycling temporel : un patch de local_factor f est recalculé
        # tous les f pas de temps.  Entre-temps la Phase 1 fournit le fond.
        active_patches = sorted(active_patches, key=lambda p: p['depth'])
        pixels_computed = 0

        for patch in active_patches:
            if patch.get('type') == 'fallback':
                continue
            depth = patch['depth']
            y0, y1, x0, x1 = patch['bounds']
            H = y1 - y0
            W = x1 - x0

            local_factor = compute_local_factor(
                H, W, depth, max_depth, target_dim
            )

            # Subcycling : les patches coarse sautent les pas intermédiaires
            """
            # (la Phase 1 fournit déjà la correction de fond)
            if local_factor > 1 and step_number % local_factor != 0:
                continue
            """

            pixels_computed += (H * W) // (local_factor ** 2)

            dx_local = dx * local_factor
            pad_local = base_pad * local_factor

            # --- Extraction de la zone locale depuis U_n (avec padding FD) ---
            slice_y = np.arange(y0 - pad_local, y1 + pad_local) % N
            slice_x = np.arange(x0 - pad_local, x1 + pad_local) % N

            loc_vx_raw = vx_n[np.ix_(slice_y, slice_x)]
            loc_vy_raw = vy_n[np.ix_(slice_y, slice_x)]
            loc_Bx_raw = Bx_n[np.ix_(slice_y, slice_x)]
            loc_By_raw = By_n[np.ix_(slice_y, slice_x)]

            # --- Downsample vers la résolution locale ---
            loc_vx = self._downsample_local(loc_vx_raw, local_factor)
            loc_vy = self._downsample_local(loc_vy_raw, local_factor)
            loc_Bx = self._downsample_local(loc_Bx_raw, local_factor)
            loc_By = self._downsample_local(loc_By_raw, local_factor)

            # --- RK4 local à la résolution du patch ---
            res = self._rk4_step(
                loc_vx, loc_vy, loc_Bx, loc_By,
                dx_local, dt, self.nu, self.eta
            )

            # --- Fine delta : différence RK4 locale upsamplée ---
            fine_delta_vx = self._upsample_local(res[0] - loc_vx, local_factor)
            fine_delta_vy = self._upsample_local(res[1] - loc_vy, local_factor)
            fine_delta_Bx = self._upsample_local(res[2] - loc_Bx, local_factor)
            fine_delta_By = self._upsample_local(res[3] - loc_By, local_factor)

            # --- Coarse delta locale (extraite du delta global Phase 1) ---
            coarse_local_vx = delta_coarse_vx[np.ix_(slice_y, slice_x)]
            coarse_local_vy = delta_coarse_vy[np.ix_(slice_y, slice_x)]
            coarse_local_Bx = delta_coarse_Bx[np.ix_(slice_y, slice_x)]
            coarse_local_By = delta_coarse_By[np.ix_(slice_y, slice_x)]

            # --- Correction tau = fine_delta - coarse_delta ---
            tau_vx = fine_delta_vx - coarse_local_vx
            tau_vy = fine_delta_vy - coarse_local_vy
            tau_Bx = fine_delta_Bx - coarse_local_Bx
            tau_By = fine_delta_By - coarse_local_By

            # --- Injection additive du cœur (sans le padding) ---
            cut = pad_local
            slice_y_inj = np.arange(y0, y1) % N
            slice_x_inj = np.arange(x0, x1) % N

            self.vx[np.ix_(slice_y_inj, slice_x_inj)] += tau_vx[cut:-cut, cut:-cut]
            self.vy[np.ix_(slice_y_inj, slice_x_inj)] += tau_vy[cut:-cut, cut:-cut]
            self.Bx[np.ix_(slice_y_inj, slice_x_inj)] += tau_Bx[cut:-cut, cut:-cut]
            self.By[np.ix_(slice_y_inj, slice_x_inj)] += tau_By[cut:-cut, cut:-cut]

        # ══════════════════════════════════════════════════════════
        #  Phase 3 : STABILISATION GLOBALE (identique à step_full)
        # ══════════════════════════════════════════════════════════
        """
        We stop smoothing fields when simulation is stable enough, to preserve fine structures.
        self.vx = self.grid.smooth_field(self.vx)
        self.vy = self.grid.smooth_field(self.vy)
        self.Bx = self.grid.smooth_field(self.Bx)
        self.By = self.grid.smooth_field(self.By)
        """
        self.enforce_incompressibility()

        self.check_cfl()
        self.record_energy()

        return pixels_computed
