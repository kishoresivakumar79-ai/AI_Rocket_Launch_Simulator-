# Comprehensive Physics Simulation Engine
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from config.rocket_config import RocketProfile, StageConfig, ROCKET_PRESETS
from config.environment_config import EnvironmentConfig, LAUNCH_SITES
from .atmosphere_model import AtmosphereModel
from .gravity_model import GravityModel
from .fuel_manager import FuelManager
from .trajectory import Trajectory3DOF
from .flight_controller import FlightController, FlightPhase

class SimulationEngine:
    """
    Main physics simulation engine orchestrating 3-DOF multi-stage rocket trajectories,
    gravitational field calculation, aerodynamic drag with Mach-dependent shockwaves,
    fuel mass flow, and flight phases.
    """
    def __init__(
        self,
        rocket_profile: RocketProfile,
        env_config: Optional[EnvironmentConfig] = None,
        dt: float = 0.1,
        max_time_s: float = 600.0
    ):
        self.rocket = rocket_profile
        self.env = env_config or EnvironmentConfig(site=LAUNCH_SITES["Kennedy Space Center (LC-39A)"])
        self.dt = dt
        self.max_time_s = max_time_s

        self.atmosphere = AtmosphereModel(
            sea_level_temp=self.env.sea_level_temp_k,
            sea_level_pressure=self.env.sea_level_pressure_pa
        )
        self.gravity = GravityModel(
            g0=self.env.gravity_sea_level,
            earth_radius=self.env.earth_radius_m
        )
        self.fuel_mgr = FuelManager(self.rocket)
        self.trajectory_model = Trajectory3DOF(self.atmosphere, self.gravity)
        self.flight_controller = FlightController()

    def run(self) -> Tuple[pd.DataFrame, Dict[str, Any], list]:
        """
        Executes full simulation until touchdown or max_time_s.
        Returns:
            telemetry_df: pandas DataFrame containing comprehensive state history
            summary: dictionary of flight achievements and milestones
            events: chronologically ordered mission events
        """
        time_s = 0.0
        
        # Initial state vectors on launch pad
        pos = np.array([0.0, 0.0, float(self.env.site.elevation_m)])
        vel = np.array([0.0, 0.0, 0.0])

        history = []
        crashed_or_landed = False

        while time_s <= self.max_time_s and not crashed_or_landed:
            current_mass = self.fuel_mgr.total_current_mass
            fuel_status = self.fuel_mgr.step_fuel_consumption(self.dt)
            thrust_mag = fuel_status["thrust"]

            # Attitude determination
            thrust_dir = self.trajectory_model.compute_flight_attitude(
                altitude=pos[2],
                velocity_vector=vel,
                pitch_over_alt=self.rocket.target_pitch_over_alt,
                pitch_kick_angle_deg=self.rocket.pitch_kick_angle_deg,
                final_pitch_deg=self.rocket.final_pitch_deg,
                azimuth_deg=90.0 # Standard eastward orbital launch
            )

            # Stage cross-sectional area
            curr_stage = self.fuel_mgr.current_stage
            diameter = curr_stage["diameter"] if curr_stage else 3.0
            area = math.pi * ((diameter / 2.0) ** 2)
            base_cd = curr_stage["drag_cd"] if curr_stage else 0.35

            # Wind vector with altitude shear
            wind_speed = self.env.wind_speed_mps + (self.env.wind_shear_factor * min(pos[2], 12000.0))
            wind_rad = math.radians(self.env.wind_direction_deg)
            wind_vec = np.array([wind_speed * math.cos(wind_rad), wind_speed * math.sin(wind_rad), 0.0])

            # RK4 Integration step
            new_pos, new_vel, aux = self.trajectory_model.rk4_step(
                position=pos,
                velocity=vel,
                dt=self.dt,
                mass=current_mass,
                thrust_magnitude=thrust_mag,
                thrust_direction=thrust_dir,
                cross_sectional_area=area,
                base_drag_cd=base_cd,
                wind_vector=wind_vec
            )

            # Acceleration approximation for telemetry
            acc = (new_vel - vel) / self.dt
            speed = float(np.linalg.norm(new_vel))
            downrange = float(np.sqrt(new_pos[0]**2 + new_pos[1]**2))

            # Update flight phase & events
            phase = self.flight_controller.update(
                time_s=time_s,
                position=new_pos.tolist(),
                velocity=new_vel.tolist(),
                acceleration=acc.tolist(),
                fuel_status=fuel_status,
                aux_data=aux
            )

            history.append({
                "time_s": round(time_s, 2),
                "altitude_m": round(float(new_pos[2]), 2),
                "downrange_m": round(downrange, 2),
                "x_m": round(float(new_pos[0]), 2),
                "y_m": round(float(new_pos[1]), 2),
                "z_m": round(float(new_pos[2]), 2),
                "vx_mps": round(float(new_vel[0]), 2),
                "vy_mps": round(float(new_vel[1]), 2),
                "vz_mps": round(float(new_vel[2]), 2),
                "velocity_mps": round(speed, 2),
                "velocity_kmh": round(speed * 3.6, 1),
                "mach": round(aux["mach"], 2),
                "acceleration_mps2": round(float(np.linalg.norm(acc)), 2),
                "g_load": round(aux["g_load"], 2),
                "mass_kg": round(current_mass, 1),
                "remaining_fuel_kg": round(self.fuel_mgr.total_current_fuel, 1),
                "fuel_percentage": round(self.fuel_mgr.overall_fuel_percentage, 1),
                "thrust_n": round(thrust_mag, 1),
                "drag_n": round(aux["drag_force_n"], 1),
                "dynamic_pressure_pa": round(aux["q_dyn"], 1),
                "stage_name": fuel_status["current_stage_name"],
                "phase": phase.value
            })

            # Check if rocket has reached touchdown / impact after apogee
            if self.flight_controller.apogee_logged and new_pos[2] <= 5.0 and time_s > 30.0:
                crashed_or_landed = True

            pos = new_pos
            vel = new_vel
            time_s += self.dt

        df = pd.DataFrame(history)

        summary = {
            "rocket_name": self.rocket.name,
            "launch_site": self.env.site.name,
            "max_altitude_m": round(float(df["altitude_m"].max()), 2),
            "max_altitude_km": round(float(df["altitude_m"].max() / 1000.0), 2),
            "max_velocity_mps": round(float(df["velocity_mps"].max()), 2),
            "max_velocity_mach": round(float(df["mach"].max()), 2),
            "max_g_force": round(float(df["g_load"].max()), 2),
            "max_dynamic_pressure_kpa": round(float(df["dynamic_pressure_pa"].max() / 1000.0), 2),
            "total_flight_time_s": round(float(df["time_s"].iloc[-1]), 1),
            "final_downrange_km": round(float(df["downrange_m"].iloc[-1] / 1000.0), 2),
            "initial_mass_kg": self.rocket.total_initial_mass,
            "final_mass_kg": round(float(df["mass_kg"].iloc[-1]), 1)
        }

        return df, summary, self.flight_controller.events


def simulate_rocket(
    initial_mass: float = 10000.0,
    dry_mass: float = 4000.0,
    fuel_mass: float = 6000.0,
    thrust: float = 200000.0,
    burn_time: float = 120.0,
    dt: float = 0.1
) -> np.ndarray:
    """
    Convenience function matching Section 9 basic prototype specification.
    Returns 2D numpy array with columns: [time_s, altitude_m, velocity_mps, acceleration_mps2, fuel_kg]
    """
    custom_stage = StageConfig(
        name="Main Stage",
        dry_mass=dry_mass,
        fuel_mass=fuel_mass,
        thrust=thrust,
        isp=thrust * burn_time / (fuel_mass * 9.80665) if fuel_mass > 0 else 280.0,
        burn_time=burn_time,
        diameter=1.5,
        drag_cd=0.35
    )
    profile = RocketProfile(
        name="Custom Rocket",
        description="User defined profile",
        stages=[custom_stage],
        payload_mass=0.0,
        target_pitch_over_alt=1200.0,
        pitch_kick_angle_deg=6.0,
        final_pitch_deg=60.0
    )
    sim = SimulationEngine(profile, dt=dt, max_time_s=350.0)
    df, _, _ = sim.run()
    
    # Return time, altitude, velocity, acceleration, remaining_fuel
    return df[["time_s", "altitude_m", "velocity_mps", "acceleration_mps2", "remaining_fuel_kg"]].to_numpy()
