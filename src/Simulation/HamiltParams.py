import numpy as np

class PhysicalMapper:
    def __init__(self, 
            cs=1.0, eta=0.01, Bz_guide=1.0, bias=4.0, gamma1=1.0, gamma2=2.0,
            Rm_crit=1000.0, delta_shock=5.0, d_kink=2.0, epsilon=1e-6
        ):
        """
        :param cs: Speed of sound (for Mach number)
        :param eta: Resistivity (for Reynolds number)
        :param Bz_guide: Constant toroidal guide field (for J dot B)
        """
        self.cs = cs
        self.eta = eta
        self.Bz = Bz_guide
        
        # Hyperparameters from your paper
        self.bias = bias
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.Rm_crit = Rm_crit
        self.delta_shock = delta_shock
        self.d_kink = d_kink
        self.epsilon = epsilon # Stability term

    def compute_coefficients(self, full_h, full_v, fields, threshold, AveragePhi, advanced_anomalies_enabled = False):
        # ... (Le début avec vx, vy, Rm_local, etc. reste identique) ...
        vx, vy = fields['vx'], fields['vy']
        Bx, By = fields['Bx'], fields['By']
        Jz = fields['Jz']

        v_mag = np.sqrt(vx**2 + vy**2)
        B_mag_sq = Bx**2 + By**2 + self.Bz**2
        Rm_local = v_mag / self.eta
        M_local = v_mag / self.cs
        helicity_density = Jz * self.Bz 

        # --- Hamiltonian Coefficients (VERSION PÉRIODIQUE / ROLL) ---
        # DATA VALIDITY
        H_horiz = self.bias * ((full_h- threshold)/AveragePhi)
        H_vert  = self.bias * ((full_v- threshold)/AveragePhi)

        # A. SHEAR TERM (C_ij)
        C_nodes = self.gamma1 * (1 + Rm_local / self.Rm_crit)**(-self.gamma2)
        
        #np.roll(x, -1) déplace tout vers la gauche, donc l'indice i se retrouve face à i+1
        # Horizontal : Moyenne entre (i, j) et (i, j+1) (avec wrap)
        C_horiz = (C_nodes + np.roll(C_nodes, -1, axis=1)) / 2
        
        # Vertical : Moyenne entre (i, j) et (i+1, j) (avec wrap)
        C_vert = (C_nodes + np.roll(C_nodes, -1, axis=0)) / 2

        # D. VORTICITY TERM (K_p) - Plaquettes
        # Une plaquette au site (i,j) implique les 4 coins :
        # (i,j), (i, j+1), (i+1, j), (i+1, j+1)
        J_abs = np.abs(Jz)
        
        # On additionne les 4 coins grâce aux rolls
        sum_corners = (
            J_abs +                           # Haut-Gauche (i,j)
            np.roll(J_abs, -1, axis=1) +      # Haut-Droite (i, j+1)
            np.roll(J_abs, -1, axis=0) +      # Bas-Gauche  (i+1, j)
            np.roll(J_abs, shift=(-1, -1), axis=(0, 1)) # Bas-Droite (i+1, j+1)
        )
        K_p = sum_corners / 4.0

        if advanced_anomalies_enabled:
            # B. SHOCK TERM (Delta_v) - Reste défini sur les noeuds
            is_supersonic = (M_local > 1.0).astype(float)
            Delta_v = self.delta_shock * is_supersonic * (M_local**2 - 1.0)


            # C. KINK TERM (D_ij)
            K_nodes = self.d_kink * (np.abs(helicity_density) * 1.256e-6) / (B_mag_sq + self.epsilon) 

            # Horizontal : Moyenne entre (i, j) et (i, j+1)
            D_horiz = (K_nodes + np.roll(K_nodes, -1, axis=1)) / 2
            
            # Vertical : Moyenne entre (i, j) et (i+1, j)
            D_vert  = (K_nodes + np.roll(K_nodes, -1, axis=0)) / 2

            return {
                "H_edges": (H_horiz, H_vert),
                "C_edges": (C_horiz, C_vert),
                "D_edges": (D_horiz, D_vert),
                "Delta_nodes": Delta_v,
                "K_plaquettes": K_p
            }

        return {
            "H_edges": (H_horiz, H_vert),
            "C_edges": (C_horiz, C_vert),
            "K_plaquettes": K_p
        }