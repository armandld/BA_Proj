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
        Instabilité de Kelvin-Helmholtz MHD avec Dérive (Drift).
        - Crée une couche de cisaillement (Shear Layer) qui s'enroule en vortex.
        - Ajoute une vitesse de fond pour forcer l'AMR à se déplacer rapidement.
        """
        X, Y = self.grid.X, self.grid.Y
        
        # 1. Vitesse : Profil en tanh (cisaillement doux)
        # Le fluide au centre (y ~ pi) va dans un sens, les bords dans l'autre.
        shear_width = 0.5
        v_flow = np.tanh((Y - np.pi) / shear_width)
        
        # 2. Le "Drift" (Vitesse de fond)
        # C'est CA qui rend l'anomalie difficile à attraper.
        # Tout le système se déplace à V=10 vers la droite.
        drift_velocity = 10.0 
        
        self.vx = v_flow + drift_velocity
        self.vy = np.zeros_like(X) # Pas de vitesse verticale initiale majeure

        # 3. Champ Magnétique : Stabilisant faible
        # B aligné avec le flux pour créer une tension magnétique
        self.Bx = 0.1 * np.ones_like(X) 
        self.By = np.zeros_like(X)

        # 4. Perturbation (L'étincelle)
        # On ajoute du bruit sur vy pour déclencher l'instabilité
        # Sinon les lignes resteraient droites éternellement.
        noise_amplitude = 0.1
        perturbation = noise_amplitude * np.sin(X) * np.exp(-((Y - np.pi)**2) / (shear_width**2))
        
        self.vy += perturbation
        
        # Projection pour nettoyer la divergence initiale
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
        LIMIT = 500.0 # Valeur arbitraire de sécurité
        
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
        # Update simplifiée pour l'AMR (Euler + Clamping Fort)
        rhs_vx, rhs_vy, rhs_Bx, rhs_By = self.compute_rhs(self.vx, self.vy, self.Bx, self.By)
        
        self.vx += self.dt * rhs_vx
        self.vy += self.dt * rhs_vy
        self.Bx += self.dt * rhs_Bx
        self.By += self.dt * rhs_By
        
        self.vx = self.grid.smooth_field(self.vx)
        self.vy = self.grid.smooth_field(self.vy)
        self.Bx = self.grid.smooth_field(self.Bx)
        self.By = self.grid.smooth_field(self.By)
        
        self.enforce_incompressibility()