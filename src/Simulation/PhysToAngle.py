import numpy as np

class AngleMapper:
    def __init__(self, v0=1.0, B0=1.0, w_shock=2.0, w_shear=1.0):
        """
        Args:
            v0: Characteristic velocity (Alfvén speed).
            B0: Characteristic magnetic field.
            w_shock: Weighting for compressive shocks (usually higher).
            w_shear: Weighting for shear/kink instability.
        """
        self.v0 = v0
        self.B0 = B0
        self.w_shock = w_shock
        self.w_shear = w_shear

    def _compute_filtered_flux(self, d_norm_v, d_tang_v, d_norm_B, d_tang_B):
        """
        Helper to apply the Shock-Diode Logic.
        - d_norm: Difference along the normal vector (Candidate for Shock)
        - d_tang: Difference perpendicular to normal (Candidate for Shear)
        """
        # 1. Shock Component (Compression Only)
        # If d_norm is negative, it means compression (Danger). 
        # We flip the sign to make it positive magnitude.
        # If d_norm is positive (Expansion), max(0, -pos) becomes 0.
        shock_v = np.maximum(0, -d_norm_v) 
        shock_B = np.maximum(0, -d_norm_B) # Magnetic compression
        
        total_shock = np.sqrt(shock_v**2 + shock_B**2)

        # 2. Shear Component (Always Dangerous)
        # Sliding motion is unstable regardless of direction (positive or negative)
        shear_v = np.abs(d_tang_v)
        shear_B = np.abs(d_tang_B)
        
        total_shear = np.sqrt(shear_v**2 + shear_B**2)

        # 3. Weighted Combination
        return np.sqrt((self.w_shock * total_shock)**2 + (self.w_shear * total_shear)**2)

    def compute_stress_flux(self, physics_state):
        """
        Computes the MHD Stress (Phi) on all graph edges.
        
        Args:
            physics_state (dict): Output from MHDSolver.get_state_for_quantum()
                                  Contains {'vx', 'vy', 'Bx', 'By', 'Jz'} as 2D arrays.
                                  
        Returns:
            dict: {
                'phi_horizontal': 2D array of fluxes on horizontal edges.
                'phi_vertical':   2D array of fluxes on vertical edges.
            }
        """
        # 1. Unpack and Normalize the State Vectors
        # We create a 3D tensor U of shape (4, N, N)
        # 4 channels: [vx/v0, vy/v0, Bx/B0, By/B0]
        vx = physics_state['vx'] / self.v0
        vy = physics_state['vy'] / self.v0
        Bx = physics_state['Bx'] / self.B0
        By = physics_state['By'] / self.B0
        
        # 2. Compute Differences for Horizontal Edges (Right Neighbors)
        # We compare pixel (i,j) with (i, j+1) and with (i, j-1) using roll(-1, axis=1)
        # Note: roll handles periodicity automatically (Torus topology)
        diff_h_vx_right = np.roll(vx, -1, axis=1) - vx
        diff_h_vy_right = np.roll(vy, -1, axis=1) - vy
        diff_h_Bx_right = np.roll(Bx, -1, axis=1) - Bx
        diff_h_By_right = np.roll(By, -1, axis=1) - By

        diff_h_vx_left = np.roll(vx, 1, axis=1) - vx
        diff_h_vy_left = np.roll(vy, 1, axis=1) - vy
        diff_h_Bx_left = np.roll(Bx, 1, axis=1) - Bx
        diff_h_By_left = np.roll(By, 1, axis=1) - By
        
        # Calculate Norm: sqrt(dvx^2 + dvy^2 + dBx^2 + dBy^2)
        phi_h_right = self._compute_filtered_flux(diff_h_vx_right, diff_h_vy_right, diff_h_Bx_right, diff_h_By_right)
        phi_h_left  = self._compute_filtered_flux(diff_h_vx_left, diff_h_vy_left, diff_h_Bx_left, diff_h_By_left)
        
        # Take the maximum danger from either side
        phi_h = np.maximum(phi_h_left, phi_h_right)
        
        # 3. Compute Differences for Vertical Edges (Bottom Neighbors)
        # We compare pixel (i,j) with (i+1, j) and with (i-1,j) using roll(-1, axis=0)
        diff_v_vx_bottom = np.roll(vx, -1, axis=0) - vx
        diff_v_vy_bottom = np.roll(vy, -1, axis=0) - vy
        diff_v_Bx_bottom = np.roll(Bx, -1, axis=0) - Bx
        diff_v_By_bottom = np.roll(By, -1, axis=0) - By

        diff_v_vx_high = np.roll(vx, 1, axis=0) - vx
        diff_v_vy_high = np.roll(vy, 1, axis=0) - vy
        diff_v_Bx_high = np.roll(Bx, 1, axis=0) - Bx
        diff_v_By_high = np.roll(By, 1, axis=0) - By
        
        # Calculate Norm
        phi_v_bottom = self._compute_filtered_flux(diff_v_vy_bottom, diff_v_vx_bottom, diff_v_By_bottom, diff_v_Bx_bottom)
        phi_v_top    = self._compute_filtered_flux(diff_v_vy_high, diff_v_vx_high, diff_v_By_high, diff_v_Bx_high)

        phi_v = np.maximum(phi_v_top, phi_v_bottom)

        return {
            'phi_horizontal': phi_h,
            'phi_vertical': phi_v
        }

    def map_to_angles(self, phi_dict, phi_dict_prev, AveragePhi, AveragePhiDev, alpha, beta, dt):
        """
        Converts Stress Flux (Phi) to Qubit Rotation Angles (Theta).
        Based on Eq (3) of your paper.
        """
        # We process both edge sets
        theta_h = self._activation_function_theta(phi_dict['phi_horizontal'], alpha, AveragePhi)
        theta_v = self._activation_function_theta(phi_dict['phi_vertical'], alpha, AveragePhi)
        if phi_dict_prev is None:
            return theta_h, theta_v,np.zeros_like(theta_h), np.zeros_like(theta_v)
        else:
            psi_h = self._activation_function_psi(phi_dict_prev['phi_horizontal'], phi_dict['phi_horizontal'], beta, dt, AveragePhiDev)
            psi_v = self._activation_function_psi(phi_dict_prev['phi_vertical'], phi_dict['phi_vertical'], beta, dt, AveragePhiDev)

        return theta_h, theta_v, psi_h, psi_v

    def _activation_function_theta(self, phi, alpha, AveragePhi):
        """
        Eq (3): theta = pi/2 * (1 + tanh(alpha * phi))
        Normalizes flux to [0, pi] range for Qubit rotation.
        """
        # Note: We assume Phi_crit = 0 and Phi_max = 1 for simplicity in Toy Model,
        # or that alpha absorbs the scaling.
        PHYSICAL_SILENCE = 1e-4 

        # 2. Safety Clamp
        denominator = max(AveragePhi, PHYSICAL_SILENCE)

        return 2.0 * np.arctan(alpha * phi / denominator)

    def _activation_function_psi(self, phi_prev, phi, beta, dt, AveragePhiDev):
        """
        Alternative activation function mapping psi to [0, 1].
        Example: Sigmoid-like mapping.
        """
        if phi_prev is None or AveragePhiDev is None:
            return np.zeros_like(phi)
        return np.pi * np.tanh(beta * ((phi - phi_prev) * dt/AveragePhiDev))