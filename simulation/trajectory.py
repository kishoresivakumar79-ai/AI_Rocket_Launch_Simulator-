# 3-DOF Rocket Flight Trajectory and Dynamics Model
import math
import numpy as np
from typing import Tuple, Dict, Any
from .atmosphere_model import AtmosphereModel
from .gravity_model import GravityModel

class Trajectory3DOF:
    """
    Three Degrees of Freedom (3-DOF) translation dynamics model for rocket flight.
    Coordinates:
    - x: Downrange displacement (East / launch azimuth) [m]
    - y: Cross-range displacement (North) [m]
    - z: Altitude above reference sea level [m]
    """
    def __init__(self, atmosphere: AtmosphereModel, gravity: GravityModel):
        self.atmosphere = atmosphere
        self.gravity = gravity

    def compute_flight_attitude(
        self,
        altitude: float,
        velocity_vector: np.ndarray,
        pitch_over_alt: float = 1500.0,
        pitch_kick_angle_deg: float = 8.0,
        final_pitch_deg: float = 75.0,
        azimuth_deg: float = 90.0  # 90 degrees = East
    ) -> np.ndarray:
        """
        Determines unit thrust attitude vector.
        - Below pitch_over_alt: Pure vertical ascent (zenith = [0, 0, 1]).
        - Above pitch_over_alt: Smooth aerospace gravity-turn pitch program
          pitching down toward the target azimuth (East).
        """
        if altitude < pitch_over_alt:
            return np.array([0.0, 0.0, 1.0])

        # Pitch program as altitude climbs from pitch_over_alt to 90 km
        h_turn_span = 90000.0 - pitch_over_alt
        alt_progress = max(0.0, min(1.0, (altitude - pitch_over_alt) / h_turn_span))
        
        # Smooth pitch growth (degrees from vertical)
        pitch_deg = pitch_kick_angle_deg + (final_pitch_deg - pitch_kick_angle_deg) * (alt_progress ** 0.5)

        theta_rad = math.radians(pitch_deg)
        az_rad = math.radians(azimuth_deg)

        # Convert spherical pitch/azimuth to Cartesian direction vector
        ux = math.sin(theta_rad) * math.sin(az_rad)
        uy = math.sin(theta_rad) * math.cos(az_rad)
        uz = math.cos(theta_rad)

        vec = np.array([ux, uy, uz])
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else np.array([0.0, 0.0, 1.0])

    def compute_derivatives(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        mass: float,
        thrust_magnitude: float,
        thrust_direction: np.ndarray,
        cross_sectional_area: float,
        base_drag_cd: float,
        wind_vector: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Calculates d(pos)/dt = v and d(v)/dt = a, along with instantaneous aerodynamic telemetry.
        """
        alt = position[2]
        
        # Apparent wind / relative airspeed
        v_rel = velocity - wind_vector
        rel_speed = float(np.linalg.norm(v_rel))
        
        # Atmospheric telemetry
        mach = self.atmosphere.get_mach_number(rel_speed, alt)
        drag_cd = self.atmosphere.get_drag_coefficient(base_drag_cd, mach)
        q_dyn = self.atmosphere.get_dynamic_pressure(rel_speed, alt)
        
        # Aerodynamic Drag Force vector opposes relative airspeed
        if rel_speed > 1e-4:
            drag_force_mag = q_dyn * drag_cd * cross_sectional_area
            drag_force = -drag_force_mag * (v_rel / rel_speed)
        else:
            drag_force_mag = 0.0
            drag_force = np.zeros(3)

        # Thrust Force vector
        thrust_force = thrust_magnitude * thrust_direction

        # Gravity Force vector
        g_vec = self.gravity.get_gravity_vector(position[0], position[1], position[2])
        gravity_force = mass * g_vec

        # Net Force and Acceleration (Newton's 2nd Law)
        net_force = thrust_force + drag_force + gravity_force
        acceleration = net_force / max(1.0, mass)

        g_load = float(np.linalg.norm(acceleration) / 9.80665)

        aux_data = {
            "mach": mach,
            "q_dyn": q_dyn,
            "drag_cd": drag_cd,
            "drag_force_n": float(drag_force_mag),
            "thrust_force_n": float(thrust_magnitude),
            "g_vec_mag": float(np.linalg.norm(g_vec)),
            "g_load": g_load
        }

        return velocity, acceleration, aux_data

    def rk4_step(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        dt: float,
        mass: float,
        thrust_magnitude: float,
        thrust_direction: np.ndarray,
        cross_sectional_area: float,
        base_drag_cd: float,
        wind_vector: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Performs a 4th-Order Runge-Kutta numerical integration step for maximum physical accuracy.
        """
        k1_v, k1_a, aux = self.compute_derivatives(
            position, velocity, mass, thrust_magnitude, thrust_direction,
            cross_sectional_area, base_drag_cd, wind_vector
        )

        # k2
        p2 = position + 0.5 * dt * k1_v
        v2 = velocity + 0.5 * dt * k1_a
        k2_v, k2_a, _ = self.compute_derivatives(
            p2, v2, mass, thrust_magnitude, thrust_direction,
            cross_sectional_area, base_drag_cd, wind_vector
        )

        # k3
        p3 = position + 0.5 * dt * k2_v
        v3 = velocity + 0.5 * dt * k2_a
        k3_v, k3_a, _ = self.compute_derivatives(
            p3, v3, mass, thrust_magnitude, thrust_direction,
            cross_sectional_area, base_drag_cd, wind_vector
        )

        # k4
        p4 = position + dt * k3_v
        v4 = velocity + dt * k3_a
        k4_v, k4_a, _ = self.compute_derivatives(
            p4, v4, mass, thrust_magnitude, thrust_direction,
            cross_sectional_area, base_drag_cd, wind_vector
        )

        new_pos = position + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        new_vel = velocity + (dt / 6.0) * (k1_a + 2*k2_a + 2*k3_a + k4_a)

        # Prevent underground penetration if falling back to surface
        if new_pos[2] < 0.0:
            new_pos[2] = 0.0
            if new_vel[2] < 0:
                new_vel = np.zeros(3)

        return new_pos, new_vel, aux
