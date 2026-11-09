# Synthetic Simulation Telemetry Dataset Generator for AI Training
import os
import random
import numpy as np
import pandas as pd
from typing import Optional

from config.rocket_config import RocketProfile, StageConfig
from simulation.physics_engine import SimulationEngine

def generate_flight_dataset(
    num_samples: int = 150,
    output_path: str = "d:/project folder/AI_Rocket_Launch_Simulator/data/simulation_data.csv",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a diverse dataset of rocket configurations, runs physics simulations,
    and records input parameters alongside simulated ground-truth flight outcomes.
    """
    np.random.seed(seed)
    random.seed(seed)

    records = []
    print(f"Generating {num_samples} simulation runs for AI model training...")

    for i in range(num_samples):
        # Sample realistic aerospace parameters
        dry_mass = float(np.random.uniform(2500, 20000))        # kg
        fuel_ratio = float(np.random.uniform(1.2, 3.5))         # fuel to dry mass ratio
        fuel_mass = float(dry_mass * fuel_ratio)
        total_mass = dry_mass + fuel_mass

        # Liftoff TWR between 1.25 and 2.2
        twr = float(np.random.uniform(1.25, 2.2))
        thrust = float(total_mass * 9.80665 * twr)              # N

        isp = float(np.random.uniform(255, 330))                # s
        diameter = float(np.random.uniform(1.0, 3.8))           # m
        drag_cd = float(np.random.uniform(0.28, 0.45))
        pitch_angle = float(np.random.uniform(4.0, 14.0))       # deg
        burn_time = (fuel_mass * isp * 9.80665) / thrust        # s

        stage = StageConfig(
            name="Test Booster",
            dry_mass=dry_mass,
            fuel_mass=fuel_mass,
            thrust=thrust,
            isp=isp,
            burn_time=burn_time,
            diameter=diameter,
            drag_cd=drag_cd
        )

        profile = RocketProfile(
            name=f"Synth_Rocket_{i+1}",
            description="Synthetic training profile",
            stages=[stage],
            payload_mass=float(np.random.uniform(100, 2000)),
            target_pitch_over_alt=float(np.random.uniform(1000, 2500)),
            pitch_kick_angle_deg=pitch_angle,
            final_pitch_deg=float(np.random.uniform(50, 80))
        )

        try:
            sim = SimulationEngine(profile, dt=0.2, max_time_s=360.0)
            df, summary, _ = sim.run()

            # Record feature inputs and target outputs
            records.append({
                "dry_mass_kg": round(dry_mass, 1),
                "fuel_mass_kg": round(fuel_mass, 1),
                "initial_total_mass_kg": round(total_mass, 1),
                "thrust_n": round(thrust, 1),
                "liftoff_twr": round(twr, 2),
                "isp_s": round(isp, 1),
                "burn_time_s": round(burn_time, 1),
                "diameter_m": round(diameter, 2),
                "drag_cd": round(drag_cd, 3),
                "pitch_kick_deg": round(pitch_angle, 1),
                # Ground-truth targets calculated by physics engine
                "max_altitude_m": summary["max_altitude_m"],
                "max_altitude_km": summary["max_altitude_km"],
                "max_velocity_mps": summary["max_velocity_mps"],
                "max_velocity_mach": summary["max_velocity_mach"],
                "max_g_force": summary["max_g_force"],
                "max_q_kpa": summary["max_dynamic_pressure_kpa"],
                "total_flight_time_s": summary["total_flight_time_s"],
                "downrange_km": summary["final_downrange_km"]
            })
        except Exception as e:
            continue

    dataset_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset_df.to_csv(output_path, index=False)
    print(f"Generated {len(dataset_df)} valid simulation runs. Saved to {output_path}")
    return dataset_df

if __name__ == "__main__":
    generate_flight_dataset(num_samples=120)

