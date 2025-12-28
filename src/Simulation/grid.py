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

    # --- Décodage VQA ---
    # Dans PeriodicGrid (grid.py)

    def decode_refinement_patches(self, vqa_probs, low_thresh=0.6, high_thresh=0.85, padding=1):
        """
        Retourne des patchs avec un facteur de raffinement dynamique (x2 ou x4)
        selon l'intensité de la probabilité quantique.
        """
        # 1. Reconstruire la carte de probabilité 2D (Max des arêtes H et V pour chaque case)
        # On veut savoir: "Quelle est la proba max qu'il y ait un souci dans cette cellule ?"
        num_edges = self.N * self.N
        probs_h = np.array(vqa_probs[:num_edges]).reshape((self.N, self.N))
        probs_v = np.array(vqa_probs[num_edges:]).reshape((self.N, self.N))
        
        # Carte de chaleur combinée (Cellule (i,j) prend le max de ses 4 arêtes)
        # Pour simplifier: on prend le max local H et V
        prob_map = np.maximum(probs_h, probs_v) 
        # (Note: une implémentation parfaite propagerait les 4 voisins, mais ceci suffit)

        # 2. Masque binaire large (Tout ce qui mérite attention)
        mask_attention = prob_map > low_thresh
        
        effective_padding = padding
        if self.N <= 8:
            effective_padding = 0
        
        if effective_padding > 0:
            mask_attention = binary_dilation(mask_attention, iterations=effective_padding)
            
        # 3. Clustering
        labeled_array, num_features = label(mask_attention)
        slices = find_objects(labeled_array)
        
        patches = []
        for sl in slices:
            # Extraction des coordonnées
            start_i, end_i = sl[0].start, sl[0].stop
            start_j, end_j = sl[1].start, sl[1].stop
            height = end_i - start_i
            width = end_j - start_j
            size = max(height, width)
            
            # --- INTELLIGENCE VQA ICI ---
            # On regarde la probabilité MAX à l'intérieur de ce cluster spécifique
            # Pour décider de la gravité
            cluster_probs = prob_map[sl]
            # On applique le masque du label pour ne pas prendre les zéros autour
            # (Simplification: on prend juste le max du rectangle bounding box)
            max_p_in_cluster = np.max(cluster_probs)
            
            # Décision du Facteur
            if max_p_in_cluster >= high_thresh:
                factor = 4  # URGENCE ABSOLUE
            else:
                factor = 2  # Turbulence standard
            
            patches.append({
                'i_start': start_i,
                'j_start': start_j,
                'width': size,
                'factor': factor  # Nouvelle info
            })
        print(f"PATCHES DETECTED: {len(patches)}")
        print(patches)
        return patches