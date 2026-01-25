import numpy as np
from scipy.ndimage import zoom

class MHDSolver:
    """
    Solveur MHD 2D Stabilisé pour Proof-of-Concept.
    Sécurités : 
    - RK2 (Heun)
    - Filtre Spatial (Shapiro)
    - Gradient Clamping (Anti-Explosion)
    """
    def __init__(self, grid, dt=1e-4, Re=100, Rm=100):
        self.grid = grid
        self.dt = dt
        # On augmente la viscosité pour la stabilité (Re=100 est plus sûr que 1000)
        self.nu = 1.0 / Re       
        self.eta = 1.0 / Rm      
        
        self.vx = np.zeros((grid.N, grid.N))
        self.vy = np.zeros((grid.N, grid.N))
        self.Bx = np.zeros((grid.N, grid.N))
        self.By = np.zeros((grid.N, grid.N))

    def init_orszag_tang(self):
        X, Y = self.grid.X, self.grid.Y
        self.vx = -np.sin(Y)
        self.vy =  np.sin(X)
        self.Bx = -np.sin(Y)
        self.By =  np.sin(2 * X)
        self.enforce_incompressibility()

    def init_kelvin_helmholtz(self):
        """
        Version Périodique Correcte.
        Deux bandes de cisaillement pour assurer la continuité aux bords (Haut/Bas).
        """
        X, Y = self.grid.X, self.grid.Y
        
        # Double couche :
        # Bande centrale (y ~ pi) va à droite (+1)
        # Bandes haute/basse (y ~ 0 et 2pi) vont à gauche (-1)
        # Cela assure que v(0) == v(2pi)
        shear_width = 0.5
        v_flow = np.tanh((Y - np.pi/2) / shear_width) - np.tanh((Y - 3*np.pi/2) / shear_width) - 1.0
        
        drift_velocity = 0.5
        self.vx = v_flow + drift_velocity
        self.vy = np.zeros_like(X)

        # Perturbation sur les DEUX interfaces
        noise_amplitude = 0.1
        # On perturbe autour de pi/2 et 3pi/2
        perturbation = noise_amplitude * np.sin(X) * (
            np.exp(-((Y - np.pi/2)**2) / (shear_width**2)) + 
            np.exp(-((Y - 3*np.pi/2)**2) / (shear_width**2))
        )
        
        self.vy += perturbation
        
        # Stabilisation Magnétique
        self.Bx = 0.1 * np.ones_like(X)
        self.By = np.zeros_like(X)

        self.enforce_incompressibility()

    def get_fluxes(self):
        grad_By_x, _ = self.grid.grad(self.By)
        _, grad_Bx_y = self.grid.grad(self.Bx)
        Jz = grad_By_x - grad_Bx_y
        return { 'vx': self.vx, 'vy': self.vy, 'Bx': self.Bx, 'By': self.By, 'Jz': Jz }

    def enforce_incompressibility(self):
        self.vx, self.vy = self.grid.project_divergence_free(self.vx, self.vy)
        self.Bx, self.By = self.grid.project_divergence_free(self.Bx, self.By)

    # --- NOYAU PHYSIQUE SÉCURISÉ ---
    def compute_rhs(self, vx, vy, Bx, By):
        """Calcule les dérivées avec CLAMPING pour éviter les NaN"""
        
        # 1. Gradients
        g_vx_x, g_vx_y = self.grid.grad(vx)
        g_vy_x, g_vy_y = self.grid.grad(vy)
        g_Bx_x, g_Bx_y = self.grid.grad(Bx)
        g_By_x, g_By_y = self.grid.grad(By)
        
        # 2. Diffusion
        diff_vx = self.nu * self.grid.laplacian(vx)
        diff_vy = self.nu * self.grid.laplacian(vy)
        diff_Bx = self.eta * self.grid.laplacian(Bx)
        diff_By = self.eta * self.grid.laplacian(By)

        # 3. Calcul des Termes Non-Linéaires
        # On utilise np.clip pour éviter que (v * grad) ne produise l'infini lors d'un choc
        LIMIT = 1e7 # Valeur arbitraire de sécurité
        
        def safe_mult(a, b):
            return np.clip(a * b, -LIMIT, LIMIT)

        adv_v_x = safe_mult(vx, g_vx_x) + safe_mult(vy, g_vx_y)
        adv_v_y = safe_mult(vx, g_vx_y) + safe_mult(vy, g_vy_y)
        adv_B_x = safe_mult(vx, g_Bx_x) + safe_mult(vy, g_Bx_y)
        adv_B_y = safe_mult(vx, g_By_x) + safe_mult(vy, g_By_y)

        lorentz_x = safe_mult(Bx, g_Bx_x) + safe_mult(By, g_Bx_y)
        lorentz_y = safe_mult(Bx, g_By_x) + safe_mult(By, g_By_y)
        stretch_x = safe_mult(Bx, g_vx_x) + safe_mult(By, g_vx_y)
        stretch_y = safe_mult(Bx, g_vy_x) + safe_mult(By, g_vy_y)
        
        # Assemblage
        rhs_vx = -adv_v_x + lorentz_x + diff_vx
        rhs_vy = -adv_v_y + lorentz_y + diff_vy
        rhs_Bx = -adv_B_x + stretch_x + diff_Bx
        rhs_By = -adv_B_y + stretch_y + diff_By
        
        # Sécurité ultime sur les dérivées
        return (np.clip(rhs_vx, -LIMIT, LIMIT), 
                np.clip(rhs_vy, -LIMIT, LIMIT), 
                np.clip(rhs_Bx, -LIMIT, LIMIT), 
                np.clip(rhs_By, -LIMIT, LIMIT))

    # --- SIMULATION STABILISÉE ---
    def step_full(self):
        # Predictor (RK2)
        k1_vx, k1_vy, k1_Bx, k1_By = self.compute_rhs(self.vx, self.vy, self.Bx, self.By)
        
        vx_guess = self.vx + self.dt * k1_vx
        vy_guess = self.vy + self.dt * k1_vy
        Bx_guess = self.Bx + self.dt * k1_Bx
        By_guess = self.By + self.dt * k1_By
        
        # Corrector
        k2_vx, k2_vy, k2_Bx, k2_By = self.compute_rhs(vx_guess, vy_guess, Bx_guess, By_guess)
        
        self.vx += (self.dt / 2.0) * (k1_vx + k2_vx)
        self.vy += (self.dt / 2.0) * (k1_vy + k2_vy)
        self.Bx += (self.dt / 2.0) * (k1_Bx + k2_Bx)
        self.By += (self.dt / 2.0) * (k1_By + k2_By)
        
        # Filtre de lissage (Nettoyage post-calcul)
        self.vx = self.grid.smooth_field(self.vx)
        self.vy = self.grid.smooth_field(self.vy)
        self.Bx = self.grid.smooth_field(self.Bx)
        self.By = self.grid.smooth_field(self.By)
        
        self.enforce_incompressibility()

    def step_layered(self, patches, max_depth):
        """
        Exécute un pas de temps adaptatif où la résolution dépend de la profondeur relative.
        
        Logique inversée :
        - depth == max_depth : Scale 1.0 (Résolution Native) -> Précision Max
        - depth == 0         : Scale faible (Gros pixels) -> Calcul Rapide
        """
        # Copies pour éviter les effets de bord immédiats pendant la boucle
        new_vx = self.vx.copy()
        new_vy = self.vy.copy()
        new_Bx = self.Bx.copy()
        new_By = self.By.copy()

        for patch in patches:
            y0, y1, x0, x1 = patch['bounds']
            
            # --- 1. DEFINE DIMENSIONS (Missing in your code) ---
            h_patch = y1 - y0  # <--- ADD THIS
            w_patch = x1 - x0  # <--- ADD THIS

            # --- 2. THEORETICAL SCALE ---
            relative_level = max_depth - patch['depth']
            theoretical_scale = 1.0 / (2.0 ** relative_level)

            # --- 3. SAFETY CHECK (2-Pixel Rule) ---
            # We need at least 2 pixels to calculate a gradient.
            # If h_patch is 16, we need scale >= 2/16 (0.125).
            # We take the max of width/height requirements to be safe.
            min_scale_h = 2.0 / h_patch
            min_scale_w = 2.0 / w_patch
            min_safe_scale = max(min_scale_h, min_scale_w)

            # --- 4. APPLY SCALE ---
            # Use theoretical scale UNLESS it crashes the math
            scale = max(theoretical_scale, min_safe_scale)
            
            # Extraction
            loc_vx = self.vx[y0:y1, x0:x1]
            loc_vy = self.vy[y0:y1, x0:x1]
            loc_Bx = self.Bx[y0:y1, x0:x1]
            loc_By = self.By[y0:y1, x0:x1]
            
            # 1. Downsampling (Si on n'est pas à la résolution max)
            if scale < 1.0:
                # On réduit la taille du tableau pour accélérer le calcul physique
                # Exemple : un patch 64x64 devient 32x32 si scale=0.5
                w_vx = zoom(loc_vx, scale, order=1)
                w_vy = zoom(loc_vy, scale, order=1)
                w_Bx = zoom(loc_Bx, scale, order=1)
                w_By = zoom(loc_By, scale, order=1)
            else:
                # Résolution native (depth == max_depth)
                w_vx, w_vy, w_Bx, w_By = loc_vx, loc_vy, loc_Bx, loc_By

            # 2. Calcul Physique
            # Le scale est passé pour corriger les gradients (dx "physique" est plus grand quand scale < 1)
            rhs_vx, rhs_vy, rhs_Bx, rhs_By = self._local_rhs(w_vx, w_vy, w_Bx, w_By, scale)

            # 3. Time Integration (Euler)
            w_vx += self.dt * rhs_vx
            w_vy += self.dt * rhs_vy
            w_Bx += self.dt * rhs_Bx
            w_By += self.dt * rhs_By

            # 4. Upsampling & Injection
            if scale < 1.0:
                # On recrée la taille cible exacte pour éviter les erreurs d'arrondi de 'zoom'
                target_shape = (y1 - y0, x1 - x0)
                
                # On calcule le facteur de zoom inverse nécessaire pour retrouver la taille du patch
                curr_shape = w_vx.shape
                zoom_back = (target_shape[0]/curr_shape[0], target_shape[1]/curr_shape[1])
                
                new_vx[y0:y1, x0:x1] = zoom(w_vx, zoom_back, order=1)
                new_vy[y0:y1, x0:x1] = zoom(w_vy, zoom_back, order=1)
                new_Bx[y0:y1, x0:x1] = zoom(w_Bx, zoom_back, order=1)
                new_By[y0:y1, x0:x1] = zoom(w_By, zoom_back, order=1)
            else:
                new_vx[y0:y1, x0:x1] = w_vx
                new_vy[y0:y1, x0:x1] = w_vy
                new_Bx[y0:y1, x0:x1] = w_Bx
                new_By[y0:y1, x0:x1] = w_By

        # Mise à jour globale
        self.vx = new_vx
        self.vy = new_vy
        self.Bx = new_Bx
        self.By = new_By
        
        # Nettoyage des artefacts de raccordement
        self.vx = self.grid.smooth_field(self.vx)
        self.vy = self.grid.smooth_field(self.vy)
        self.Bx = self.grid.smooth_field(self.Bx)
        self.By = self.grid.smooth_field(self.By)
        
        self.enforce_incompressibility()

    def _local_rhs(self, vx, vy, Bx, By, scale):
        """
        Calcule la physique sur un patch potentiellement redimensionné.
        Correction physique : Gradient_local = Gradient_numpy * scale
        """
        # Si scale = 0.5 (image réduite de moitié), la distance entre deux index
        # représente 2 unités d'espace d'origine. La pente est donc plus douce.
        # Il faut multiplier le gradient numpy par scale pour retrouver la pente physique correcte.
        
        def grad(f):
            g = np.gradient(f)
            return g[1] * scale, g[0] * scale # dx, dy

        def laplacian(f):
            # Le laplacien est une dérivée seconde, donc scale au carré
            return  (np.gradient(np.gradient(f, axis=0), axis=0) + 
                     np.gradient(np.gradient(f, axis=1), axis=1)) * (scale**2)

        # --- Dérivées Spatiales ---
        g_vx_x, g_vx_y = grad(vx)
        g_vy_x, g_vy_y = grad(vy)
        g_Bx_x, g_Bx_y = grad(Bx)
        g_By_x, g_By_y = grad(By)
        
        diff_vx = self.nu * laplacian(vx)
        diff_vy = self.nu * laplacian(vy)
        diff_Bx = self.eta * laplacian(Bx)
        diff_By = self.eta * laplacian(By)

        # --- Termes Non-Linéaires (Sécurisés) ---
        LIMIT = 500.0
        def safe_mult(a, b): return np.clip(a * b, -LIMIT, LIMIT)

        # Advection (v . grad)
        adv_v_x = safe_mult(vx, g_vx_x) + safe_mult(vy, g_vx_y)
        adv_v_y = safe_mult(vx, g_vx_y) + safe_mult(vy, g_vy_y)
        adv_B_x = safe_mult(vx, g_Bx_x) + safe_mult(vy, g_Bx_y)
        adv_B_y = safe_mult(vx, g_By_x) + safe_mult(vy, g_By_y)

        # Termes Magnétiques (Lorentz + Stretching)
        lorentz_x = safe_mult(Bx, g_Bx_x) + safe_mult(By, g_Bx_y)
        lorentz_y = safe_mult(Bx, g_By_x) + safe_mult(By, g_By_y)
        stretch_x = safe_mult(Bx, g_vx_x) + safe_mult(By, g_vx_y)
        stretch_y = safe_mult(Bx, g_vy_x) + safe_mult(By, g_vy_y)
        
        # Assemblage RHS
        rhs_vx = -adv_v_x + lorentz_x + diff_vx
        rhs_vy = -adv_v_y + lorentz_y + diff_vy
        rhs_Bx = -adv_B_x + stretch_x + diff_Bx
        rhs_By = -adv_B_y + stretch_y + diff_By

        return rhs_vx, rhs_vy, rhs_Bx, rhs_By