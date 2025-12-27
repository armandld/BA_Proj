import numpy as np
from .grid import PeriodicGrid

class MHDSolver:
    """
    Solveur MHD 2D Incompressible Visco-Résistif.
    Équations:
      dv/dt = -(v.grad)v + (B.grad)B + nu*Lap(v)
      dB/dt = -(v.grad)B + (B.grad)v + eta*Lap(B)
    """
    def __init__(self, grid: PeriodicGrid, dt=1e-3, Re=1000, Rm=1000):
        self.grid = grid
        self.dt = dt
        self.nu = 1.0 / Re       # Viscosité
        self.eta = 1.0 / Rm      # Résistivité
        
        # Champs Physiques (Vitesse et Magnétique)
        # Initialisés à 0
        self.vx = np.zeros((grid.N, grid.N))
        self.vy = np.zeros((grid.N, grid.N))
        self.Bx = np.zeros((grid.N, grid.N))
        self.By = np.zeros((grid.N, grid.N))

    def init_orszag_tang(self):
        """
        Configure les conditions initiales du Vortex d'Orszag-Tang.
        Source: Eq 16 du papier.
        v = (-sin y, sin x)
        B = (-sin y, sin 2x)
        """
        X, Y = self.grid.X, self.grid.Y
        self.vx = -np.sin(Y)
        self.vy =  np.sin(X)
        self.Bx = -np.sin(Y)
        self.By =  np.sin(2 * X)

    def time_step(self):
        """Avance d'un pas de temps (Euler Explicite pour simplicité)"""
        # 1. Calcul des gradients nécessaires
        grad_vx_x, grad_vx_y = self.grid.grad(self.vx)
        grad_vy_x, grad_vy_y = self.grid.grad(self.vy)
        grad_Bx_x, grad_Bx_y = self.grid.grad(self.Bx)
        grad_By_x, grad_By_y = self.grid.grad(self.By)
        
        # 2. Termes Advectifs (v . grad)
        adv_v_x = self.vx * grad_vx_x + self.vy * grad_vx_y
        adv_v_y = self.vx * grad_vy_x + self.vy * grad_vy_y
        
        adv_B_x = self.vx * grad_Bx_x + self.vy * grad_Bx_y
        adv_B_y = self.vx * grad_By_x + self.vy * grad_By_y

        # 3. Termes de Lorentz / Stretching (B . grad)
        str_v_x = self.Bx * grad_vx_x + self.By * grad_vx_y
        str_v_y = self.Bx * grad_vy_x + self.By * grad_vy_y
        
        str_B_x = self.Bx * grad_Bx_x + self.By * grad_Bx_y
        str_B_y = self.Bx * grad_By_x + self.By * grad_By_y

        # 4. Termes de Diffusion (Laplacien)
        lap_vx = self.grid.laplacian(self.vx)
        lap_vy = self.grid.laplacian(self.vy)
        lap_Bx = self.grid.laplacian(self.Bx)
        lap_By = self.grid.laplacian(self.By)

        # 5. Mise à jour temporelle (Navier-Stokes + Induction)
        # dv/dt = - Advection + Lorentz + Diffusion
        self.vx += self.dt * (-adv_v_x + str_B_x + self.nu * lap_vx)
        self.vy += self.dt * (-adv_v_y + str_B_y + self.nu * lap_vy)
        
        # dB/dt = - Advection + Stretching + Diffusion
        self.Bx += self.dt * (-adv_B_x + str_v_x + self.eta * lap_Bx)
        self.By += self.dt * (-adv_B_y + str_v_y + self.eta * lap_By)

    def get_fluxes(self):
        """
        Prépare les données pour le Mapping Quantique (VQA).
        Retourne Bx, By (Flux) et la Densité de Courant Jz (Curl B).
        """
        grad_By_x, _ = self.grid.grad(self.By)
        _, grad_Bx_y = self.grid.grad(self.Bx)
        Jz = grad_By_x - grad_Bx_y # J = rot B
        
        return {
            'vx': self.vx,
            'vy': self.vy,
            'Bx': self.Bx,
            'By': self.By,
            'Jz': Jz
        }