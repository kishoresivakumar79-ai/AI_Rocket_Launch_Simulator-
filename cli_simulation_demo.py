"""
CLI Flight Telemetry & Simulation Demo
Runs an end-to-end mission directly in the terminal with ASCII metrics.
"""
import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.rocket_config import ROCKET_PRESETS
from config.environment_config import LAUNCH_SITES, EnvironmentConfig
from simulation.physics_engine import SimulationEngine
from ai_engine.prediction import FlightOutcomePredictor

def run_cli_demo():
    print("=" * 65)
    print("      🚀 AI ROCKET LAUNCH SIMULATOR — CLI TELEMETRY DEMO")
    print("=" * 65)

    profile = ROCKET_PRESETS["Falcon 9 Medium-Lift (2-Stage)"]
    env = EnvironmentConfig(site=LAUNCH_SITES["Kennedy Space Center (LC-39A)"])

    print(f"\n[VEHICLE]    : {profile.name}")
    print(f"[LAUNCH PAD] : {env.site.name} (Lat: {env.site.latitude_deg}°, Elev: {env.site.elevation_m}m)")
    print(f"[MASS SPEC]  : Dry {profile.total_dry_mass:,.0f} kg | Propellant: {profile.total_fuel_mass:,.0f} kg | Payload: {profile.payload_mass:,.0f} kg")
    print(f"[STAGES]     : {len(profile.stages)} Stages")

    # Fast AI Pre-Flight Inference
    print("\n--- 🧠 STEP 1: AI PRE-FLIGHT OUTCOME INFERENCE ---")
    predictor = FlightOutcomePredictor()
    ai_preds = predictor.predict(profile)
    print(f"Predicted Apogee Altitude : {ai_preds['predicted_apogee_km']:.2f} km")
    print(f"Predicted Max Velocity    : {ai_preds['predicted_max_velocity_mps']:,.1f} m/s")
    print(f"Predicted Max-Q Pressure  : {ai_preds['predicted_max_q_kpa']:.2f} kPa")
    print(f"Predicted Flight Duration : {ai_preds['predicted_flight_duration_s']:.1f} s")

    # High-Fidelity Physics Engine Integration
    print("\n--- 🛰️ STEP 2: RUNNING 3-DOF RK4 NUMERICAL INTEGRATION ---")
    sim = SimulationEngine(profile, env, dt=0.5, max_time_s=600.0)
    df, summary, events = sim.run()

    print("\n[FLIGHT EVENTS TIMELINE]")
    for ev in events:
        print(f"  T+{ev['time_s']:>5.1f}s | {ev['event']:<22s} | Alt: {ev['altitude_m']:>8,.0f} m | Vel: {ev['velocity_mps']:>6,.1f} m/s | Downrange: {ev['downrange_km']:>6.1f} km")

    print("\n--- 🎯 STEP 3: MEASURED PHYSICAL PERFORMANCE ---")
    print(f"Max Altitude (Apogee)    : {summary['max_altitude_km']:.2f} km")
    print(f"Max Velocity             : {summary['max_velocity_mps']:,.1f} m/s (Mach {summary['max_velocity_mach']:.2f})")
    print(f"Max Dynamic Pressure (Q) : {summary['max_dynamic_pressure_kpa']:.2f} kPa")
    print(f"Peak G-Load              : {summary['max_g_force']:.2f} G")
    print(f"Total Flight Duration    : {summary['total_flight_time_s']:.1f} s")
    print(f"Downrange Distance       : {summary['final_downrange_km']:.2f} km")

    # AI Anomaly Evaluation
    print("\n--- 🛡️ STEP 4: AI TELEMETRY ANOMALY DETECTION ---")
    anomaly = predictor.detect_anomaly(summary)
    print(f"Assessment   : {anomaly['status']}")
    print(f"Anomaly Score: {anomaly['anomaly_score']:.3f}")
    print(f"Diagnostics  : {anomaly['notes']}")

    print("\n" + "=" * 65)
    print("Flight complete. Launch interactive dashboard with: streamlit run app.py")
    print("=" * 65)

if __name__ == "__main__":
    run_cli_demo()

