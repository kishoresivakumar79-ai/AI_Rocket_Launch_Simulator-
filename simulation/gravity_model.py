# Gravitational Field Model
import numpy as np

class GravityModel:
    """
    Newtonian gravity model accounting for inverse-square altitude decay.
    g(h) = g0 * (R_earth / (R_earth + h))^2
    """
    def __init__(self, g0: float = 9.80665, earth_radius: float = 6371000.0):
        self.g0 = g0
        self.earth_radius = earth_radius

    def get_gravity(self, altitude_m: float) -> float:
        """Calculate local acceleration due to gravity at given altitude in m/s^2."""
        h = max(0.0, float(altitude_m))
        ratio = self.earth_radius / (self.earth_radius + h)
        return self.g0 * (ratio ** 2)

    def get_gravity_vector(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Computes 3D gravity vector directed towards Earth's center.
        Assuming z is altitude above sea level, x and y are horizontal ground displacements.
        """
        r_mag = np.sqrt(x**2 + y**2 + (self.earth_radius + max(0.0, z))**2)
        g_mag = self.g0 * (self.earth_radius / r_mag)**2
        
        # Unit vector pointing towards Earth center (0, 0, -R_earth)
        rx = -x
        ry = -y
        rz = -(self.earth_radius + max(0.0, z))
        
        return g_mag * (np.array([rx, ry, rz]) / r_mag)

