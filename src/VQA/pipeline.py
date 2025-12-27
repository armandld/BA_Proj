import numpy as np

class QuantumMapper:
    def __init__(self, v0=1.0, B0=1.0):
        """
        Args:
            v0: Characteristic velocity (e.g., Alfven speed) for normalization.
            B0: Characteristic magnetic field for normalization.
        """
        self.v0 = v0
        self.B0 = B0

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
        # We compare pixel (i,j) with (i, j+1) using roll(-1, axis=1)
        # Note: roll handles periodicity automatically (Torus topology)
        diff_h_vx = np.roll(vx, -1, axis=1) - vx
        diff_h_vy = np.roll(vy, -1, axis=1) - vy
        diff_h_Bx = np.roll(Bx, -1, axis=1) - Bx
        diff_h_By = np.roll(By, -1, axis=1) - By
        
        # Calculate Norm: sqrt(dvx^2 + dvy^2 + dBx^2 + dBy^2)
        phi_h = np.sqrt(diff_h_vx**2 + diff_h_vy**2 + diff_h_Bx**2 + diff_h_By**2)

        # 3. Compute Differences for Vertical Edges (Bottom Neighbors)
        # We compare pixel (i,j) with (i+1, j) using roll(-1, axis=0)
        diff_v_vx = np.roll(vx, -1, axis=0) - vx
        diff_v_vy = np.roll(vy, -1, axis=0) - vy
        diff_v_Bx = np.roll(Bx, -1, axis=0) - Bx
        diff_v_By = np.roll(By, -1, axis=0) - By
        
        # Calculate Norm
        phi_v = np.sqrt(diff_v_vx**2 + diff_v_vy**2 + diff_v_Bx**2 + diff_v_By**2)

        return {
            'phi_horizontal': phi_h,
            'phi_vertical': phi_v
        }

    def map_to_angles(self, phi_dict, phi_dict_prev, alpha, beta, dt):
        """
        Converts Stress Flux (Phi) to Qubit Rotation Angles (Theta).
        Based on Eq (3) of your paper.
        """
        # We process both edge sets
        theta_h = self._activation_function_theta(phi_dict['phi_horizontal'], alpha)
        theta_v = self._activation_function_theta(phi_dict['phi_vertical'], alpha)
        psi_h = self._activation_function_psi(phi_dict_prev['phi_horizontal'], phi_dict['phi_horizontal'], beta, dt)
        psi_v = self._activation_function_psi(phi_dict_prev['phi_vertical'], phi_dict['phi_vertical'], beta, dt)

        return theta_h, theta_v

    def _activation_function_theta(self, phi, alpha):
        """
        Eq (3): theta = pi/2 * (1 + tanh(alpha * phi))
        Normalizes flux to [0, pi] range for Qubit rotation.
        """
        # Note: We assume Phi_crit = 0 and Phi_max = 1 for simplicity in Toy Model,
        # or that alpha absorbs the scaling.
        return (np.pi / 2) * (1 + np.tanh(alpha * phi))
    
    def _activation_function_psi(self, phi_prev, phi, beta, dt):
        """
        Alternative activation function mapping psi to [0, 1].
        Example: Sigmoid-like mapping.
        """
        if phi_prev is None:
            return np.zeros_like(phi)
        return np.pi * np.tanh(beta * (phi - phi_prev * dt))