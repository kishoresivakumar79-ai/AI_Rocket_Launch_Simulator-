# Machine Learning Model Training Pipeline
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from .dataset_generator import generate_flight_dataset

FEATURE_COLUMNS = [
    "dry_mass_kg",
    "fuel_mass_kg",
    "initial_total_mass_kg",
    "thrust_n",
    "liftoff_twr",
    "isp_s",
    "burn_time_s",
    "diameter_m",
    "drag_cd",
    "pitch_kick_deg"
]

TARGET_COLUMNS = [
    "max_altitude_km",
    "max_velocity_mps",
    "max_q_kpa",
    "total_flight_time_s"
]

def train_flight_models(
    dataset_path: str = "d:/project folder/AI_Rocket_Launch_Simulator/data/simulation_data.csv",
    models_dir: str = "d:/project folder/AI_Rocket_Launch_Simulator/ai_engine/models"
):
    """
    Trains Random Forest and Gradient Boosting models on the physics simulation dataset.
    Also trains an Isolation Forest for real-time telemetry anomaly detection.
    """
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Generating now...")
        df = generate_flight_dataset(num_samples=100, output_path=dataset_path)
    else:
        df = pd.read_csv(dataset_path)

    print(f"Loaded {len(df)} flight records for training.")
    os.makedirs(models_dir, exist_ok=True)

    X = df[FEATURE_COLUMNS]
    metrics_report = {}
    trained_models = {}

    X_train, X_test, df_train, df_test = train_test_split(X, df, test_size=0.2, random_state=42)

    # 1. Train Target Regressors (Random Forest & Gradient Boosting)
    for target in TARGET_COLUMNS:
        y_train = df_train[target]
        y_test = df_test[target]

        # Use Random Forest as primary predictor
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))

        metrics_report[target] = {
            "r2_score": round(r2, 4),
            "rmse": round(rmse, 2),
            "mae": round(mae, 2)
        }
        trained_models[target] = model
        
        # Save individual target model
        target_path = os.path.join(models_dir, f"rf_model_{target}.joblib")
        joblib.dump(model, target_path)

    # 2. Train AI Telemetry Anomaly Detection Model (Isolation Forest)
    # Learns normal flight boundary distributions across mass, altitude, velocity, and g-force
    anomaly_detector = IsolationForest(
        n_estimators=80,
        contamination=0.03,
        random_state=42
    )
    # Fit anomaly detector on key flight telemetry features
    anomaly_features = df[["initial_total_mass_kg", "thrust_n", "max_altitude_km", "max_velocity_mps", "max_g_force"]]
    anomaly_detector.fit(anomaly_features)
    joblib.dump(anomaly_detector, os.path.join(models_dir, "anomaly_detector.joblib"))

    # Save metadata and feature names
    metadata = {
        "features": FEATURE_COLUMNS,
        "targets": TARGET_COLUMNS,
        "metrics": metrics_report,
        "num_samples": len(df)
    }
    joblib.dump(metadata, os.path.join(models_dir, "model_metadata.joblib"))

    print("\n=== Model Training Complete ===")
    for target, m in metrics_report.items():
        print(f"Target: {target:20s} | R^2: {m['r2_score']:.3f} | RMSE: {m['rmse']:.2f} | MAE: {m['mae']:.2f}")

    return trained_models, metrics_report

if __name__ == "__main__":
    train_flight_models()

