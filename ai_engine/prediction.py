# Machine Learning Inference & Telemetry Anomaly Detection
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from config.rocket_config import RocketProfile
from .train_model import FEATURE_COLUMNS, TARGET_COLUMNS, train_flight_models

class FlightOutcomePredictor:
    """
    Fast AI Flight Performance Predictor.
    Provides real-time machine-learned estimations of apogee, top speed,
    dynamic pressure, and mission duration before firing the physics engine.
    """
    def __init__(self, models_dir: str = "d:/project folder/AI_Rocket_Launch_Simulator/ai_engine/models"):
        self.models_dir = models_dir
        self.models = {}
        self.metadata = {}
        self.anomaly_detector = None
        self._load_or_train()

    def _load_or_train(self):
        metadata_path = os.path.join(self.models_dir, "model_metadata.joblib")
        if not os.path.exists(metadata_path):
            print("Models not found on disk. Initiating training pipeline...")
            self.models, _ = train_flight_models(models_dir=self.models_dir)
            if os.path.exists(metadata_path):
                self.metadata = joblib.load(metadata_path)
            anomaly_path = os.path.join(self.models_dir, "anomaly_detector.joblib")
            if os.path.exists(anomaly_path):
                self.anomaly_detector = joblib.load(anomaly_path)
        else:
            self.metadata = joblib.load(metadata_path)
            for target in TARGET_COLUMNS:
                model_path = os.path.join(self.models_dir, f"rf_model_{target}.joblib")
                if os.path.exists(model_path):
                    self.models[target] = joblib.load(model_path)
            anomaly_path = os.path.join(self.models_dir, "anomaly_detector.joblib")
            if os.path.exists(anomaly_path):
                self.anomaly_detector = joblib.load(anomaly_path)

    def extract_features(self, profile: RocketProfile) -> pd.DataFrame:
        """Extracts standard input features matching training data."""
        stage1 = profile.stages[0]
        total_mass = profile.total_initial_mass
        dry_mass = profile.total_dry_mass
        fuel_mass = profile.total_fuel_mass
        thrust = stage1.thrust
        g0 = 9.80665
        twr = thrust / max(1.0, total_mass * g0)
        isp = stage1.isp
        burn_time = (fuel_mass * isp * g0) / max(1.0, thrust)

        row = {
            "dry_mass_kg": dry_mass,
            "fuel_mass_kg": fuel_mass,
            "initial_total_mass_kg": total_mass,
            "thrust_n": thrust,
            "liftoff_twr": twr,
            "isp_s": isp,
            "burn_time_s": burn_time,
            "diameter_m": stage1.diameter,
            "drag_cd": stage1.drag_cd,
            "pitch_kick_deg": profile.pitch_kick_angle_deg
        }
        return pd.DataFrame([row])

    def predict(self, profile: RocketProfile) -> Dict[str, Any]:
        """Runs fast AI regression inference across all target flight metrics."""
        feat_df = self.extract_features(profile)
        predictions = {}

        for target in TARGET_COLUMNS:
            if target in self.models:
                pred_val = float(self.models[target].predict(feat_df)[0])
                predictions[target] = round(pred_val, 2)
            else:
                predictions[target] = None

        return {
            "predicted_apogee_km": predictions.get("max_altitude_km", 0.0),
            "predicted_max_velocity_mps": predictions.get("max_velocity_mps", 0.0),
            "predicted_max_q_kpa": predictions.get("max_q_kpa", 0.0),
            "predicted_flight_duration_s": predictions.get("total_flight_time_s", 0.0),
            "input_features": feat_df.iloc[0].to_dict()
        }

    def detect_anomaly(self, flight_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uses trained Isolation Forest to detect if flight outcome or telemetry
        demonstrates anomalous behavior (e.g., severe drag deceleration, engine failure).
        """
        if self.anomaly_detector is None:
            return {"is_anomaly": False, "score": 0.0, "status": "NOMINAL (Detector not loaded)"}

        features_df = pd.DataFrame([{
            "initial_total_mass_kg": flight_summary.get("initial_mass_kg", 10000.0),
            "thrust_n": flight_summary.get("initial_mass_kg", 10000.0) * 9.81 * 1.5,
            "max_altitude_km": flight_summary.get("max_altitude_km", 50.0),
            "max_velocity_mps": flight_summary.get("max_velocity_mps", 1000.0),
            "max_g_force": flight_summary.get("max_g_force", 3.0)
        }])

        score = float(self.anomaly_detector.decision_function(features_df)[0])
        pred = int(self.anomaly_detector.predict(features_df)[0])
        is_anomaly = (pred == -1)

        status_msg = "ANOMALY DETECTED" if is_anomaly else "NOMINAL FLIGHT PROFILE"
        explanation = []
        if flight_summary.get("max_g_force", 0) > 6.0:
            explanation.append("High structural G-force load (> 6.0 G)")
        if flight_summary.get("max_dynamic_pressure_kpa", 0) > 65.0:
            explanation.append("Elevated aerodynamic dynamic pressure (> 65 kPa)")
        if flight_summary.get("max_altitude_km", 0) < 5.0:
            explanation.append("Premature trajectory termination / low apogee")

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(score, 3),
            "status": status_msg,
            "notes": ", ".join(explanation) if explanation else "Telemetry conforms to learned physics envelopes."
        }
