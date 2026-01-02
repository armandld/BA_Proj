import numpy as np
from scipy.ndimage import zoom, label, find_objects, binary_dilation

class PeriodicGrid:
    """
    Représente une grille spatiale 2D périodique [0, L] x [0, L].
    Gère les dérivées spatiales et le raffinement (AMR).
    """
    def __init__(self, resolution_N, length_L=2*np.pi):
        self.N = resolution_N
        self.L = length_L
        self.dx = length_L / resolution_N
        
        # Coordonnées (pour l'initialisation physique)
        x = np.linspace(0, length_L, resolution_N, endpoint=False)
        y = np.linspace(0, length_L, resolution_N, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
    
    def resolution(self):
        return self.N

    def smooth_field(self, f):
        """
        Applique un léger lissage pour éliminer les instabilités de grille (bruit pixel).
        Essentiel pour les différences finies à haut Reynolds.
        """
        # Lissage en X (0.25 - 0.5 - 0.25)
        f = 0.5 * f + 0.25 * (np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0))
        # Lissage en Y (0.25 - 0.5 - 0.25)
        f = 0.5 * f + 0.25 * (np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1))
        return f

    def grad(self, f):
        """Calcul du Gradient (df/dx, df/dy) via Différences Finies Centrées"""
        df_dx = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * self.dx)
        df_dy = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * self.dx)
        return df_dx, df_dy

    def div(self, fx, fy):
        """Calcul de la Divergence (dfx/dx + dfy/dy)"""
        dfx_dx = (np.roll(fx, -1, axis=0) - np.roll(fx, 1, axis=0)) / (2 * self.dx)
        dfy_dy = (np.roll(fy, -1, axis=1) - np.roll(fy, 1, axis=1)) / (2 * self.dx)
        return dfx_dx + dfy_dy

    def laplacian(self, f):
        """Calcul du Laplacien (d2f/dx2 + d2f/dy2)"""
        d2f_dx2 = (np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0)) / (self.dx**2)
        d2f_dy2 = (np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1)) / (self.dx**2)
        return d2f_dx2 + d2f_dy2

    # --- Gestion AMR & Interpolation ---
    def extract_patch_data(self, data, i_start, j_start, width):
        indices_x = (np.arange(width) + i_start) % self.N
        indices_y = (np.arange(width) + j_start) % self.N
        return data[np.ix_(indices_x, indices_y)]

    def create_refined_grid(self, data_list, i_start, j_start, width, factor=2):
        patch_L = self.L * (width / self.N)
        new_N = width * factor
        new_grid = PeriodicGrid(resolution_N=new_N, length_L=patch_L)
        
        interpolated_data = []
        for field in data_list:
            patch_data = self.extract_patch_data(field, i_start, j_start, width)
            fine_data = zoom(patch_data, zoom=factor, order=1)
            interpolated_data.append(fine_data)
            
        return new_grid, interpolated_data
    

    def project_divergence_free(self, vx, vy):
        """
        Projette le champ de vitesse (vx, vy) sur un espace à divergence nulle.
        Utilise la méthode spectrale (FFT) pour résoudre l'équation de Poisson:
        nabla^2 phi = div(v)
        v_final = v - nabla(phi)
        
        Cette méthode garantit que la physique reste stable et incompressible.
        """
        # 1. Passage dans l'espace de Fourier
        vx_hat = np.fft.fft2(vx)
        vy_hat = np.fft.fft2(vy)

        # 2. Création des nombres d'onde (Wave numbers kx, ky)
        # kx correspond aux fréquences spatiales le long de l'axe 0
        # ky correspond aux fréquences spatiales le long de l'axe 1
        kx = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi
        
        # Grille des fréquences (Attention à l'ordre 'ij' comme dans __init__)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')

        # 3. Calcul du carré de la norme du vecteur d'onde |k|^2
        K2 = KX**2 + KY**2
        
        # Gestion de la singularité à k=0 (la composante moyenne / DC)
        # On évite la division par 0. La moyenne du flux n'est pas modifiée par la projection.
        K2[0, 0] = 1.0 

        # 4. Calcul de la correction (Projection)
        # Dans l'espace de Fourier, div(v) devient i(kx*vx + ky*vy)
        # On cherche phi tel que -k^2 * phi_hat = div_hat
        div_hat = 1j * KX * vx_hat + 1j * KY * vy_hat
        phi_hat = div_hat / K2  # Résolution de Poisson

        # 5. Soustraction du gradient de phi (correction)
        # v_new = v - grad(phi)
        # grad(phi) devient i*k * phi_hat
        vx_hat -= 1j * KX * phi_hat
        vy_hat -= 1j * KY * phi_hat

        # 6. Forcer le mode 0 (moyenne) à rester inchangé ou nul pour la correction
        # (Optionnel mais propre mathématiquement)
        # vx_hat[0,0] et vy_hat[0,0] sont conservés tels quels par la soustraction ci-dessus 
        # car KX[0,0] = 0.

        # 7. Retour dans l'espace réel
        # On prend la partie réelle car des erreurs d'arrondi machine peuvent créer une partie imaginaire minuscule
        vx_new = np.real(np.fft.ifft2(vx_hat))
        vy_new = np.real(np.fft.ifft2(vy_hat))

        return vx_new, vy_new