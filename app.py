"""
AI Rocket Launch Simulator - Mission Control Dashboard
Combines Aerospace Physics + 3-DOF Trajectory + Machine Learning Predictions + 3D Visualization
"""
import os
import sys
import time
import pandas as pd
import numpy as np
import streamlit as st

# Add simulator root to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.rocket_config import ROCKET_PRESETS, RocketProfile, StageConfig
from config.environment_config import LAUNCH_SITES, EnvironmentConfig
from simulation.physics_engine import SimulationEngine
import streamlit.components.v1 as components
from visualization.trajectory_3d import create_3d_trajectory_figure
from visualization.altitude_graph import create_altitude_figure
from visualization.velocity_graph import create_velocity_figure
from visualization.fuel_graph import create_fuel_mass_figure
from visualization.rocket_animation import (
    create_gauge_indicator,
    create_playback_snapshot,
    generate_animated_rocket_html
)
from ai_engine.prediction import FlightOutcomePredictor
from ai_engine.model_evaluation import create_feature_importance_figure
from reports.report_generator import FlightReportGenerator

# Page setup
st.set_page_config(
    page_title="AI Rocket Launch Simulator — Mission Control",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Mission Control Dark CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0c0f17;
        color: #e0e6ed;
    }
    .metric-card {
        background: linear-gradient(135deg, #161b26 0%, #1c2333 100%);
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #00e5ff;
        font-family: 'Courier New', monospace;
    }
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 1px;
    }
    .phase-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        letter-spacing: 1px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_ai_engine():
    return FlightOutcomePredictor()


ai_predictor = load_ai_engine()

# Initialize session state
if "telemetry_df" not in st.session_state:
    st.session_state.telemetry_df = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "events" not in st.session_state:
    st.session_state.events = []
if "ai_preds" not in st.session_state:
    st.session_state.ai_preds = None
if "anomaly_report" not in st.session_state:
    st.session_state.anomaly_report = None
if "launched" not in st.session_state:
    st.session_state.launched = False

# Sidebar: Rocket & Mission Configuration
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1517976487507-5b3b4a4507d3?auto=format&fit=crop&w=400&q=80", use_container_width=True)
    st.title("🎛️ Mission Control")
    st.markdown("Configure launch vehicle, launch pad environment, and flight parameters.")

    preset_name = st.selectbox(
        "Rocket Architecture Preset",
        options=list(ROCKET_PRESETS.keys()) + ["Custom Rocket"]
    )

    if preset_name != "Custom Rocket":
        preset = ROCKET_PRESETS[preset_name]
        st.info(f"**{preset.name}**\n\n{preset.description}")
        
        stages_count = len(preset.stages)
        st.write(f"**Stages:** {stages_count} | **Payload:** {preset.payload_mass:,.0f} kg")

        # Allow slight parameter adjustments
        with st.expander("Adjust Parameters", expanded=False):
            adj_fuel = st.slider("Fuel Multiplier", 0.5, 1.5, 1.0, 0.05)
            adj_thrust = st.slider("Thrust Multiplier", 0.7, 1.4, 1.0, 0.05)
            pitch_kick_angle = st.slider("Pitch Kick Angle (deg)", 2.0, 15.0, float(preset.pitch_kick_angle_deg))

        # Reconstructed profile
        tuned_stages = []
        for s in preset.stages:
            tuned_stages.append(StageConfig(
                name=s.name,
                dry_mass=s.dry_mass,
                fuel_mass=s.fuel_mass * adj_fuel,
                thrust=s.thrust * adj_thrust,
                isp=s.isp,
                burn_time=s.burn_time * adj_fuel / adj_thrust,
                diameter=s.diameter,
                drag_cd=s.drag_cd
            ))
        active_profile = RocketProfile(
            name=preset.name,
            description=preset.description,
            stages=tuned_stages,
            payload_mass=preset.payload_mass,
            target_pitch_over_alt=preset.target_pitch_over_alt,
            pitch_kick_angle_deg=pitch_kick_angle,
            final_pitch_deg=preset.final_pitch_deg
        )
    else:
        st.subheader("Custom Vehicle Spec")
        c_dry = st.number_input("Dry Mass (kg)", value=5000.0, step=500.0)
        c_fuel = st.number_input("Fuel Mass (kg)", value=12000.0, step=1000.0)
        c_thrust = st.number_input("Engine Thrust (N)", value=320000.0, step=10000.0)
        c_isp = st.number_input("Specific Impulse Isp (s)", value=290.0, step=5.0)
        c_diam = st.number_input("Core Diameter (m)", value=1.8, step=0.1)
        c_cd = st.slider("Drag Coefficient Cd", 0.2, 0.6, 0.35, 0.01)
        c_pitch = st.slider("Gravity Turn Pitch Kick (deg)", 3.0, 15.0, 8.0)
        c_payload = st.number_input("Payload Mass (kg)", value=500.0, step=100.0)

        burn_time_calc = (c_fuel * c_isp * 9.80665) / c_thrust
        custom_stage = StageConfig(
            name="Single Stage Booster",
            dry_mass=c_dry,
            fuel_mass=c_fuel,
            thrust=c_thrust,
            isp=c_isp,
            burn_time=burn_time_calc,
            diameter=c_diam,
            drag_cd=c_cd
        )
        active_profile = RocketProfile(
            name="Custom Experimental Vehicle",
            description="User configured orbital prototype",
            stages=[custom_stage],
            payload_mass=c_payload,
            target_pitch_over_alt=1200.0,
            pitch_kick_angle_deg=c_pitch,
            final_pitch_deg=75.0
        )

    # Launch site & environmental selection
    st.subheader("🌍 Launch Environment")
    site_name = st.selectbox("Spaceport / Launch Site", list(LAUNCH_SITES.keys()))
    site_obj = LAUNCH_SITES[site_name]
    wind_spd = st.slider("Surface Wind Speed (m/s)", 0.0, 25.0, 5.0, 1.0)
    wind_dir = st.slider("Wind Direction (deg azimuth)", 0, 360, 90, 15)

    env_config = EnvironmentConfig(
        site=site_obj,
        wind_speed_mps=wind_spd,
        wind_direction_deg=wind_dir
    )

    # Pre-Launch AI Fast Prediction
    st.subheader("🧠 Pre-Launch AI Prediction")
    ai_pre_preds = ai_predictor.predict(active_profile)
    col_p1, col_p2 = st.columns(2)
    col_p1.metric("AI Pred Apogee", f"{ai_pre_preds['predicted_apogee_km']:.1f} km")
    col_p2.metric("AI Pred Max Vel", f"{ai_pre_preds['predicted_max_velocity_mps']:,.0f} m/s")

    # Launch button
    st.divider()
    launch_btn = st.button("🚀 INITIATE LAUNCH SEQUENCE", type="primary", use_container_width=True)
    reset_btn = st.button("🔄 Reset Simulation", use_container_width=True)

if reset_btn:
    st.session_state.telemetry_df = None
    st.session_state.summary = None
    st.session_state.events = []
    st.session_state.ai_preds = None
    st.session_state.anomaly_report = None
    st.session_state.launched = False
    st.rerun()

# Run simulation if launch pressed or already cached
if launch_btn or (st.session_state.telemetry_df is None and not st.session_state.launched):
    with st.spinner("🚀 Ignition in progress... Integrating 3-DOF differential motion & aerodynamics..."):
        sim = SimulationEngine(
            rocket_profile=active_profile,
            env_config=env_config,
            dt=0.2,
            max_time_s=500.0
        )
        df, summary, events = sim.run()
        ai_preds = ai_predictor.predict(active_profile)
        anomaly = ai_predictor.detect_anomaly(summary)

        st.session_state.telemetry_df = df
        st.session_state.summary = summary
        st.session_state.events = events
        st.session_state.ai_preds = ai_preds
        st.session_state.anomaly_report = anomaly
        st.session_state.launched = True

df = st.session_state.telemetry_df
summary = st.session_state.summary
events = st.session_state.events
ai_preds = st.session_state.ai_preds
anomaly = st.session_state.anomaly_report

# Top Aerospace Mission Status Banner
st.markdown(f"""
<div style="background: #111827; border-left: 5px solid #00e5ff; padding: 12px 20px; border-radius: 6px; margin-bottom: 20px;">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h2 style="margin: 0; color: #fff; font-size: 24px;">🚀 {summary['rocket_name']} — Flight Telemetry</h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 13px;">Launch Site: <b>{summary['launch_site']}</b> | Trajectory: 3-DOF Gravity Turn</p>
        </div>
        <div>
            <span class="phase-badge" style="background: rgba(0, 229, 255, 0.15); color: #00e5ff; border: 1px solid #00e5ff;">
                STATUS: {df['phase'].iloc[-1]}
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Top KPI Metric Cards
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Max Altitude (Apogee)</div>
        <div class="metric-value">{summary['max_altitude_km']:.1f} <span style="font-size:14px;color:#94a3b8;">km</span></div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Max Velocity</div>
        <div class="metric-value">{summary['max_velocity_mps']:,.0f} <span style="font-size:14px;color:#94a3b8;">m/s</span></div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Peak Speed (Mach)</div>
        <div class="metric-value">M {summary['max_velocity_mach']:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Max Dynamic Pressure</div>
        <div class="metric-value">{summary['max_dynamic_pressure_kpa']:.1f} <span style="font-size:14px;color:#94a3b8;">kPa</span></div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Flight Duration</div>
        <div class="metric-value">{summary['total_flight_time_s']:.0f} <span style="font-size:14px;color:#94a3b8;">sec</span></div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Main Mission Tabs
tab_viewport, tab_3d, tab_telemetry, tab_ai, tab_report, tab_twin = st.tabs([
    "🚀 Animated Rocket Launch Viewport",
    "🛰️ 3D Trajectory & Playback",
    "📊 Telemetry Charts",
    "🧠 AI Prediction & Anomalies",
    "📋 Mission Debrief & Reports",
    "🔬 Digital Twin Comparison"
])

# Tab 1: Animated Rocket Launch Viewport
with tab_viewport:
    st.subheader(f"🚀 Visual Mission Telemetry Viewport — {summary['rocket_name']}")
    st.markdown(
        "Click **🚀 LAUNCH** inside the viewport below to begin the countdown sequence, "
        "ignite engine plumes, and watch the rocket dynamically ascend through atmospheric layers."
    )
    
    col_view, col_view_dials = st.columns([3, 1])
    with col_view:
        # Render the full animated rocket HTML5/Canvas viewport
        viewport_html = generate_animated_rocket_html(df, rocket_name=summary['rocket_name'])
        components.html(viewport_html, height=505, scrolling=False)
    
    with col_view_dials:
        st.markdown("#### 📡 Real-Time Cockpit Gauges")
        # Display peak indicators
        st.plotly_chart(create_gauge_indicator(
            value=summary['max_altitude_km'],
            title="Apogee Target",
            max_range=max(10.0, summary['max_altitude_km'] * 1.15),
            unit="km",
            color="#00e5ff"
        ), use_container_width=True)

        st.plotly_chart(create_gauge_indicator(
            value=summary['max_velocity_mps'],
            title="Peak Speed",
            max_range=max(500.0, summary['max_velocity_mps'] * 1.15),
            unit="m/s",
            color="#ff9100"
        ), use_container_width=True)

# Tab 2: 3D Trajectory & Interactive Playback Scrubber
with tab_3d:
    st.markdown("### 🛰️ 3D Spaceflight Trajectory & Vehicle Motion")
    st.info("💡 **Click '▶ PLAY 3D FLIGHT'** on the bottom left of the 3D canvas below to watch the 3D rocket vehicle travel along its trajectory!")
    col_vis, col_gauges = st.columns([3, 1])
    
    with col_vis:
        fig_3d = create_3d_trajectory_figure(df, events=events, animated=True)
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_gauges:
        st.subheader("⏱️ Flight Playback Scrubber")
        max_time = float(df["time_s"].max())
        playback_time = st.slider("Elapsed Time T+ (s)", 0.0, max_time, max_time, step=1.0)
        
        snapshot = create_playback_snapshot(df, playback_time)

        st.markdown(f"**Current Phase:** `{snapshot['phase']}`")
        st.markdown(f"**Active Stage:** `{snapshot['stage_name']}`")
        
        # Dial indicators
        st.plotly_chart(create_gauge_indicator(
            value=snapshot['altitude_m'] / 1000.0,
            title="Altitude",
            max_range=max(10.0, summary['max_altitude_km'] * 1.1),
            unit="km",
            color="#00e5ff"
        ), use_container_width=True)

        st.plotly_chart(create_gauge_indicator(
            value=snapshot['velocity_mps'],
            title="Velocity",
            max_range=max(500.0, summary['max_velocity_mps'] * 1.1),
            unit="m/s",
            color="#ff9100"
        ), use_container_width=True)

        st.plotly_chart(create_gauge_indicator(
            value=snapshot['fuel_percentage'],
            title="Fuel Remaining",
            max_range=100.0,
            unit="%",
            color="#2979ff"
        ), use_container_width=True)

# Tab 2: Telemetry Graphs
with tab_telemetry:
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.plotly_chart(create_altitude_figure(df), use_container_width=True)
        st.plotly_chart(create_fuel_mass_figure(df), use_container_width=True)
    with col_g2:
        st.plotly_chart(create_velocity_figure(df), use_container_width=True)
        
        # Real-time G-force & aerodynamic drag plot
        import plotly.express as px
        fig_forces = px.line(
            df,
            x="time_s",
            y=["thrust_n", "drag_n"],
            labels={"value": "Force (N)", "variable": "Force Channel"},
            title="Forces: Engine Thrust vs Aerodynamic Drag",
            template="plotly_dark"
        )
        st.plotly_chart(fig_forces, use_container_width=True)

# Tab 3: AI Prediction Engine & Anomaly Detection
with tab_ai:
    st.subheader("🧠 Machine Learning Flight Telemetry Analytics")
    st.markdown(
        "The AI module uses trained **Random Forest** and **Isolation Forest** models "
        "trained on hundreds of multi-stage physics simulation trajectories to estimate flight parameters "
        "and monitor telemetry health."
    )

    col_ai_left, col_ai_right = st.columns([1, 1])

    with col_ai_left:
        st.markdown("### 🎯 Physics Engine vs AI Prediction Comparison")
        comp_data = [
            {
                "Metric": "Apogee Altitude",
                "Physics Engine": f"{summary['max_altitude_km']:.2f} km",
                "AI Prediction": f"{ai_preds['predicted_apogee_km']:.2f} km",
                "Residual Delta": f"{abs(summary['max_altitude_km'] - ai_preds['predicted_apogee_km']):.2f} km"
            },
            {
                "Metric": "Max Velocity",
                "Physics Engine": f"{summary['max_velocity_mps']:,.1f} m/s",
                "AI Prediction": f"{ai_preds['predicted_max_velocity_mps']:,.1f} m/s",
                "Residual Delta": f"{abs(summary['max_velocity_mps'] - ai_preds['predicted_max_velocity_mps']):.1f} m/s"
            },
            {
                "Metric": "Max Dynamic Pressure (Max-Q)",
                "Physics Engine": f"{summary['max_dynamic_pressure_kpa']:.2f} kPa",
                "AI Prediction": f"{ai_preds['predicted_max_q_kpa']:.2f} kPa",
                "Residual Delta": f"{abs(summary['max_dynamic_pressure_kpa'] - ai_preds['predicted_max_q_kpa']):.2f} kPa"
            },
            {
                "Metric": "Mission Duration",
                "Physics Engine": f"{summary['total_flight_time_s']:.1f} s",
                "AI Prediction": f"{ai_preds['predicted_flight_duration_s']:.1f} s",
                "Residual Delta": f"{abs(summary['total_flight_time_s'] - ai_preds['predicted_flight_duration_s']):.1f} s"
            }
        ]
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

        # AI Anomaly Detection Status Box
        st.markdown("### 🛡️ AI Anomaly Detection Engine")
        is_anom = anomaly["is_anomaly"]
        box_color = "#ff1744" if is_anom else "#00e676"
        st.markdown(f"""
        <div style="background: rgba(22, 27, 38, 0.9); border: 2px solid {box_color}; border-radius: 8px; padding: 16px;">
            <h4 style="margin: 0; color: {box_color};">Status: {anomaly['status']}</h4>
            <p style="margin: 8px 0; color: #cbd5e1;"><b>Isolation Forest Score:</b> {anomaly['anomaly_score']:.3f}</p>
            <p style="margin: 0; color: #94a3b8;"><b>Diagnostics:</b> {anomaly['notes']}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_ai_right:
        st.markdown("### 📊 Model Feature Importances")
        if "max_altitude_km" in ai_predictor.models:
            rf_model = ai_predictor.models["max_altitude_km"]
            feat_fig = create_feature_importance_figure(
                rf_model,
                ai_predictor.metadata.get("features", []),
                "Apogee Altitude"
            )
            st.plotly_chart(feat_fig, use_container_width=True)
        else:
            st.info("Feature importance will appear once models are loaded.")

# Tab 4: Mission Debrief & Reports
with tab_report:
    st.subheader("📋 Flight Events Timeline & Debriefing")
    
    # Chronological timeline table
    if events:
        events_df = pd.DataFrame(events)
        st.dataframe(
            events_df.rename(columns={
                "time_s": "MET T+ (s)",
                "event": "Event Code",
                "details": "Mission Controller Notes",
                "altitude_m": "Altitude (m)",
                "velocity_mps": "Speed (m/s)",
                "downrange_km": "Downrange (km)"
            }),
            use_container_width=True
        )

    # Report Generator
    report_gen = FlightReportGenerator(summary, events, ai_preds, anomaly)
    markdown_report = report_gen.generate_markdown_report()

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "📥 Download Telemetry CSV",
            data=df.to_csv(index=False),
            file_name=f"telemetry_{summary['rocket_name'].replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_dl2:
        st.download_button(
            "📄 Download Mission Debrief Report (.md)",
            data=markdown_report,
            file_name=f"mission_report_{summary['rocket_name'].replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True
        )

    with st.expander("Preview Mission Debrief Markdown Report"):
        st.markdown(markdown_report)

# Tab 5: Digital Twin Comparison
with tab_twin:
    st.subheader("🔬 Digital Twin: Baseline vs Perturbed Variant Comparison")
    st.markdown("Simulate what happens if atmospheric density increases, engine thrust degrades by 10%, or pitch kick timing varies.")

    col_dt1, col_dt2 = st.columns(2)
    with col_dt1:
        thrust_pert = st.slider("Thrust Perturbation (%)", -30, 30, -10, 5)
        drag_pert = st.slider("Drag Perturbation (%)", -50, 100, 25, 5)
    with col_dt2:
        run_twin = st.button("Run Digital Twin Simulation", type="secondary", use_container_width=True)

    if run_twin:
        twin_stages = []
        for s in active_profile.stages:
            twin_stages.append(StageConfig(
                name=f"{s.name} (Twin)",
                dry_mass=s.dry_mass,
                fuel_mass=s.fuel_mass,
                thrust=s.thrust * (1.0 + thrust_pert / 100.0),
                isp=s.isp,
                burn_time=s.burn_time / max(0.5, 1.0 + thrust_pert / 100.0),
                diameter=s.diameter,
                drag_cd=s.drag_cd * (1.0 + drag_pert / 100.0)
            ))
        twin_profile = RocketProfile(
            name=f"{active_profile.name} (Perturbed)",
            description="Digital twin variant",
            stages=twin_stages,
            payload_mass=active_profile.payload_mass,
            target_pitch_over_alt=active_profile.target_pitch_over_alt,
            pitch_kick_angle_deg=active_profile.pitch_kick_angle_deg,
            final_pitch_deg=active_profile.final_pitch_deg
        )
        sim_twin = SimulationEngine(twin_profile, env_config, dt=0.2, max_time_s=400.0)
        df_twin, sum_twin, _ = sim_twin.run()

        # Comparative Plot
        import plotly.graph_objects as go
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(x=df["time_s"], y=df["altitude_m"]/1000.0, name="Baseline Altitude (km)", line=dict(color="#00e5ff", width=2)))
        fig_cmp.add_trace(go.Scatter(x=df_twin["time_s"], y=df_twin["altitude_m"]/1000.0, name="Digital Twin Altitude (km)", line=dict(color="#ff9100", width=2, dash="dash")))
        fig_cmp.update_layout(title="Baseline vs Digital Twin Altitude Profile", template="plotly_dark", xaxis_title="Time (s)", yaxis_title="Altitude (km)")
        st.plotly_chart(fig_cmp, use_container_width=True)

        st.metric(
            "Apogee Delta",
            f"{sum_twin['max_altitude_km']:.1f} km",
            delta=f"{sum_twin['max_altitude_km'] - summary['max_altitude_km']:.1f} km"
        )

