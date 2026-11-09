# Mission Flight Report Generator
import pandas as pd
from typing import Dict, Any, List

class FlightReportGenerator:
    """
    Generates structured mission debrief reports, flight telemetry summaries,
    and physics-vs-AI comparison analysis.
    """
    def __init__(
        self,
        summary: Dict[str, Any],
        events: List[Dict[str, Any]],
        ai_predictions: Dict[str, Any],
        anomaly_info: Dict[str, Any]
    ):
        self.summary = summary
        self.events = events
        self.ai_pred = ai_predictions
        self.anomaly = anomaly_info

    def generate_markdown_report(self) -> str:
        """Constructs a comprehensive, publication-ready mission debriefing document."""
        phys_apogee = self.summary.get("max_altitude_km", 0.0)
        ai_apogee = self.ai_pred.get("predicted_apogee_km", 0.0)
        diff_apogee = abs(phys_apogee - ai_apogee)
        pct_apogee = (diff_apogee / max(0.1, phys_apogee)) * 100.0

        phys_vel = self.summary.get("max_velocity_mps", 0.0)
        ai_vel = self.ai_pred.get("predicted_max_velocity_mps", 0.0)
        diff_vel = abs(phys_vel - ai_vel)
        pct_vel = (diff_vel / max(0.1, phys_vel)) * 100.0

        md = f"""# 🚀 AI Rocket Launch Simulator — Mission Flight Report

**Mission:** {self.summary.get('rocket_name', 'Orbital Launch')}  
**Launch Complex:** {self.summary.get('launch_site', 'Cape Canaveral')}  
**Status:** COMPLETE / MISSION DEBRIEF  

---

## 1. Flight Performance KPI Scorecard

| Telemetry Metric | Measured Physics Value | AI Pre-Launch Estimate | Error Residual (Delta) | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Max Altitude (Apogee)** | **{phys_apogee:.2f} km** | {ai_apogee:.2f} km | {diff_apogee:.2f} km | {100.0 - min(100.0, pct_apogee):.1f}% |
| **Max Velocity** | **{phys_vel:,.1f} m/s** (Mach {self.summary.get('max_velocity_mach', 0):.2f}) | {ai_vel:,.1f} m/s | {diff_vel:.1f} m/s | {100.0 - min(100.0, pct_vel):.1f}% |
| **Max Dynamic Pressure (Max-Q)** | **{self.summary.get('max_dynamic_pressure_kpa', 0):.2f} kPa** | {self.ai_pred.get('predicted_max_q_kpa', 0):.2f} kPa | {abs(self.summary.get('max_dynamic_pressure_kpa', 0) - self.ai_pred.get('predicted_max_q_kpa', 0)):.2f} kPa | - |
| **Flight Duration** | **{self.summary.get('total_flight_time_s', 0):.1f} s** | {self.ai_pred.get('predicted_flight_duration_s', 0):.1f} s | - | - |
| **Peak Acceleration** | **{self.summary.get('max_g_force', 0):.2f} G** | - | - | - |
| **Downrange Distance** | **{self.summary.get('final_downrange_km', 0):.2f} km** | - | - | - |

---

## 2. AI Telemetry Anomaly Evaluation
* **Overall Assessment:** `{self.anomaly.get('status', 'NOMINAL')}`
* **Anomaly Decision Score:** `{self.anomaly.get('anomaly_score', 0.0):.3f}`
* **Diagnostic Notes:** {self.anomaly.get('notes', 'All telemetry channels remained within nominal safety envelopes.')}

---

## 3. Mission Chronology & Event Milestones

| Elapsed Time (MET) | Event Code | Telemetry Details | Altitude | Speed | Downrange |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
        for ev in self.events:
            md += f"| **T+{ev['time_s']:.1f}s** | `{ev['event']}` | {ev['details']} | {ev['altitude_m']:,} m | {ev['velocity_mps']} m/s | {ev['downrange_km']} km |\n"

        md += """
---
*AI Rocket Launch Simulator — Aerospace Numerical Integration & Machine Learning Telemetry Engine.*
"""
        return md

