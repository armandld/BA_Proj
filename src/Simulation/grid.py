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
