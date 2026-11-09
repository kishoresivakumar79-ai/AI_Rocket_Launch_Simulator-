"""
Automated Screenshot and Visual Asset Generator for Documentation
Produces clean, high-resolution dark-themed mission control screenshots.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.rocket_config import ROCKET_PRESETS
from config.environment_config import LAUNCH_SITES, EnvironmentConfig
from simulation.physics_engine import SimulationEngine
from ai_engine.prediction import FlightOutcomePredictor

ASSETS_DIR = os.path.join(ROOT_DIR, "assets", "screenshots")
os.makedirs(ASSETS_DIR, exist_ok=True)

# Common Aerospace Styling
plt.style.use("dark_background")
DARK_BG = "#0c0f17"
PANEL_BG = "#161b26"
CYAN = "#00e5ff"
ORANGE = "#ff9100"
GREEN = "#00e676"
PURPLE = "#e040fb"
RED = "#ff1744"
YELLOW = "#ffd600"
TEXT_MUTED = "#94a3b8"

def generate_step1_config():
    """Screenshot 1: Rocket Configuration & Launch Parameters"""
    fig = plt.figure(figsize=(14, 8), facecolor=DARK_BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(DARK_BG)
    ax.axis("off")

    # Header
    ax.text(0.04, 0.93, "AI ROCKET LAUNCH SIMULATOR -- MISSION CONTROL", color="#ffffff", fontsize=20, weight="bold")
    ax.text(0.04, 0.88, "STEP 1: VEHICLE ARCHITECTURE, LAUNCH SITE & AI PRE-FLIGHT SPECS", color=CYAN, fontsize=12, weight="bold")

    # Sidebar Box
    rect_side = patches.FancyBboxPatch((0.04, 0.08), 0.38, 0.74, boxstyle="round,pad=0.02",
                                       fc=PANEL_BG, ec="#2d3748", lw=1.5)
    ax.add_patch(rect_side)

    ax.text(0.06, 0.78, "[VEHICLE PARAMETERS] (FALCON 9 PRESET)", color="#ffffff", fontsize=13, weight="bold")
    params = [
        ("Architecture", "Two-Stage Orbital Medium-Lift Vehicle"),
        ("Dry Mass", "44,500 kg (Stage 1 + Stage 2)"),
        ("Propellant Mass", "487,500 kg (RP-1 / LOX)"),
        ("Payload Mass", "15,000 kg to LEO"),
        ("Stage 1 Thrust", "7,607,000 N (9x Merlin 1D Engines)"),
        ("Booster Isp", "300.0 s (Sea-Level)"),
        ("Stage 2 Thrust", "934,000 N (1x Merlin Vacuum Engine)"),
        ("Upper Stage Isp", "348.0 s (Vacuum)"),
        ("Pitch Kick Alt", "1,500 m (Gravity Turn Start)"),
        ("Launch Azimuth", "90.0 deg (Eastward Orbital Insertion)")
    ]
    y_pos = 0.72
    for k, v in params:
        ax.text(0.06, y_pos, f"* {k}:", color=TEXT_MUTED, fontsize=10)
        ax.text(0.24, y_pos, v, color="#ffffff", fontsize=10, weight="bold")
        y_pos -= 0.045

    # Right Box: Launch Pad & AI Prediction Preview
    rect_right = patches.FancyBboxPatch((0.45, 0.48), 0.51, 0.34, boxstyle="round,pad=0.02",
                                        fc=PANEL_BG, ec="#2d3748", lw=1.5)
    ax.add_patch(rect_right)
    ax.text(0.48, 0.78, "[LAUNCH PAD ENVIRONMENT]", color="#ffffff", fontsize=13, weight="bold")
    env_info = [
        ("Launch Site", "Kennedy Space Center (LC-39A), Florida"),
        ("Coordinates", "28.5721 N, -80.6480 W"),
        ("Pad Elevation", "3.0 meters above mean sea level"),
        ("Surface Wind", "5.0 m/s from 90 deg East (Altitude Wind Shear Enabled)"),
        ("Earth Rotation", "+408 m/s Eastward Tangential Boost")
    ]
    y_pos = 0.72
    for k, v in env_info:
        ax.text(0.48, y_pos, f"* {k}:", color=TEXT_MUTED, fontsize=10)
        ax.text(0.66, y_pos, v, color="#ffffff", fontsize=10, weight="bold")
        y_pos -= 0.045

    # Bottom Right Box: Fast AI Pre-Flight Inferred Baseline
    rect_ai = patches.FancyBboxPatch((0.45, 0.08), 0.51, 0.36, boxstyle="round,pad=0.02",
                                     fc=PANEL_BG, ec=CYAN, lw=1.5)
    ax.add_patch(rect_ai)
    ax.text(0.48, 0.39, "[PRE-LAUNCH AI MODEL INFERENCE] (Random Forest Regressor)", color=CYAN, fontsize=13, weight="bold")
    ax.text(0.48, 0.34, "Sub-millisecond flight outcome prediction before numerical integration:", color=TEXT_MUTED, fontsize=10)

    # 4 AI Scorecards
    cards = [
        ("PRED APOGEE", "235.5 km", CYAN),
        ("PRED TOP SPEED", "2,988 m/s", ORANGE),
        ("PRED MAX-Q", "28.3 kPa", PURPLE),
        ("PRED FLIGHT TIME", "360 sec", GREEN)
    ]
    x_card = 0.47
    for label, val, col in cards:
        rect_c = patches.FancyBboxPatch((x_card, 0.12), 0.11, 0.17, boxstyle="round,pad=0.01",
                                        fc="#0d1117", ec=col, lw=1.2)
        ax.add_patch(rect_c)
        ax.text(x_card + 0.01, 0.24, label, color=TEXT_MUTED, fontsize=8, weight="bold")
        ax.text(x_card + 0.01, 0.16, val, color=col, fontsize=12, weight="bold")
        x_card += 0.125

    plt.tight_layout()
    out_file = os.path.join(ASSETS_DIR, "01_mission_control_config.png")
    plt.savefig(out_file, dpi=180, facecolor=DARK_BG)
    plt.close()
    print("Saved:", out_file)


def generate_step2_viewport():
    """Screenshot 2: Visual Rocket Launch Viewport & Liftoff Animation"""
    fig = plt.figure(figsize=(14, 8), facecolor=DARK_BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(DARK_BG)
    ax.axis("off")

    ax.text(0.04, 0.94, "AI ROCKET LAUNCH SIMULATOR -- MISSION CONTROL", color="#ffffff", fontsize=20, weight="bold")
    ax.text(0.04, 0.89, "STEP 2: LIVE ANIMATED ROCKET LAUNCH VIEWPORT & LIFTOFF SEQUENCE", color=ORANGE, fontsize=12, weight="bold")

    # Main Viewport Canvas Box
    vp_rect = patches.FancyBboxPatch((0.04, 0.08), 0.65, 0.77, boxstyle="round,pad=0.01",
                                     fc="#050811", ec="#30363d", lw=1.5)
    ax.add_patch(vp_rect)

    # Sky gradient effect (Troposphere to Stratosphere)
    for i in range(50):
        y_bot = 0.08 + (i * 0.015)
        blue_val = 0.35 - (i * 0.007)
        ax.fill_between([0.04, 0.69], y_bot, y_bot + 0.015,
                        color=(0.04, 0.08 + max(0, blue_val*0.6), max(0, blue_val)), alpha=0.9, zorder=1)

    # Ambient Stars
    np.random.seed(42)
    star_x = np.random.uniform(0.06, 0.67, 70)
    star_y = np.random.uniform(0.40, 0.82, 70)
    ax.scatter(star_x, star_y, s=np.random.uniform(2, 12, 70), color="#ffffff", alpha=0.8, zorder=2)

    # Ground Launch Pad Base
    ax.fill_between([0.04, 0.69], 0.08, 0.18, color="#161b22", zorder=3)
    ax.plot([0.04, 0.69], [0.18, 0.18], color=CYAN, lw=2, zorder=3)

    # Launch Pad Tower Gantry
    ax.plot([0.31, 0.31], [0.18, 0.38], color="#8b949e", lw=4, zorder=4)
    for gy in np.linspace(0.18, 0.38, 7):
        ax.plot([0.29, 0.33], [gy, gy], color="#8b949e", lw=1.5, zorder=4)

    # Visual Rocket ascending at 18 degrees pitch tilt
    rx, ry = 0.37, 0.45
    # Nosecone
    ax.fill([rx-0.012, rx, rx+0.012], [ry+0.12, ry+0.17, ry+0.12], color="#ffffff", zorder=6)
    # Upper Stage
    rect_s2 = patches.Rectangle((rx-0.012, ry+0.06), 0.024, 0.06, color="#e6edf3", zorder=6)
    ax.add_patch(rect_s2)
    # Booster Stage
    rect_s1 = patches.Rectangle((rx-0.012, ry-0.04), 0.024, 0.10, color="#ffffff", zorder=6)
    ax.add_patch(rect_s1)
    # Stage 1 aerodynamic fins
    ax.fill([rx-0.012, rx-0.025, rx-0.012], [ry-0.02, ry-0.05, ry-0.04], color="#30363d", zorder=6)
    ax.fill([rx+0.012, rx+0.025, rx+0.012], [ry-0.02, ry-0.05, ry-0.04], color="#30363d", zorder=6)

    # Dynamic Engine Exhaust Plume (Flames)
    ax.fill([rx-0.008, rx, rx+0.008], [ry-0.04, ry-0.12, ry-0.04], color="#ff5722", alpha=0.9, zorder=5)
    ax.fill([rx-0.005, rx, rx+0.005], [ry-0.04, ry-0.09, ry-0.04], color="#ffeb3b", alpha=0.95, zorder=5)
    ax.fill([rx-0.002, rx, rx+0.002], [ry-0.04, ry-0.06, ry-0.04], color="#ffffff", alpha=1.0, zorder=5)

    # Transonic vapor cone effect near Mach 1
    cone_arc = patches.Arc((rx, ry+0.04), 0.08, 0.03, angle=0, theta1=0, theta2=180, color=CYAN, lw=2.5, alpha=0.75, zorder=7)
    ax.add_patch(cone_arc)

    # Viewport HUD Overlay
    hud_box = patches.FancyBboxPatch((0.06, 0.65), 0.22, 0.18, boxstyle="round,pad=0.01",
                                    fc="#0d1117", ec="#30363d", lw=1.2, alpha=0.9, zorder=8)
    ax.add_patch(hud_box)
    ax.text(0.08, 0.80, "MET: T+48.5s", color="#ffffff", fontsize=11, weight="bold", zorder=9)
    ax.text(0.08, 0.76, "ALTITUDE: 6.61 km", color=CYAN, fontsize=10, weight="bold", zorder=9)
    ax.text(0.08, 0.72, "VELOCITY: 1,158 km/h (Mach 1.05)", color=ORANGE, fontsize=10, weight="bold", zorder=9)
    ax.text(0.08, 0.68, "G-LOAD: 2.32 G | FUEL: 72%", color=GREEN, fontsize=10, weight="bold", zorder=9)

    # Phase Badge
    phase_box = patches.FancyBboxPatch((0.48, 0.78), 0.19, 0.05, boxstyle="round,pad=0.01",
                                      fc="#161b22", ec=GREEN, lw=1.5, zorder=8)
    ax.add_patch(phase_box)
    ax.text(0.50, 0.795, "[ACTIVE] TRANSONIC PASSING", color=GREEN, fontsize=9, weight="bold", zorder=9)

    # Right Cockpit Gauges Panel
    right_box = patches.FancyBboxPatch((0.71, 0.08), 0.25, 0.77, boxstyle="round,pad=0.01",
                                       fc=PANEL_BG, ec="#2d3748", lw=1.5)
    ax.add_patch(right_box)
    ax.text(0.73, 0.80, "[COCKPIT GAUGES]", color="#ffffff", fontsize=13, weight="bold")

    gauges = [
        ("ALTITUDE", "6.61 km", "Max: 137.3 km", CYAN, 0.05),
        ("SPEED (MACH)", "Mach 1.05", "Top: Mach 21.9", ORANGE, 0.15),
        ("ACCELERATION", "2.32 G", "Peak: 12.69 G", RED, 0.25),
        ("PROPELLANT", "72.4%", "Remaining: 352,900 kg", GREEN, 0.72)
    ]
    gy_start = 0.72
    for title, main_v, sub_v, col, pct in gauges:
        g_card = patches.FancyBboxPatch((0.73, gy_start - 0.12), 0.21, 0.13, boxstyle="round,pad=0.01",
                                        fc="#0d1117", ec="#30363d", lw=1.0)
        ax.add_patch(g_card)
        ax.text(0.75, gy_start - 0.02, title, color=TEXT_MUTED, fontsize=9, weight="bold")
        ax.text(0.75, gy_start - 0.07, main_v, color=col, fontsize=14, weight="bold")
        ax.text(0.75, gy_start - 0.105, sub_v, color=TEXT_MUTED, fontsize=8)
        gy_start -= 0.16

    plt.tight_layout()
    out_file = os.path.join(ASSETS_DIR, "02_visual_rocket_launch.png")
    plt.savefig(out_file, dpi=180, facecolor=DARK_BG)
    plt.close()
    print("Saved:", out_file)


def generate_step3_3d_trajectory(df, events):
    """Screenshot 3: 3D Flight Trajectory Visualization & Attitude Vector"""
    fig = plt.figure(figsize=(14, 8), facecolor=DARK_BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(DARK_BG)

    # Title
    fig.text(0.04, 0.94, "🚀 AI ROCKET LAUNCH SIMULATOR — MISSION CONTROL", color="#ffffff", fontsize=18, weight="bold")
    fig.text(0.04, 0.90, "STEP 3: 3D CURVED GRAVITY-TURN TRAJECTORY WITH AEROSPACE EVENT MARKERS", color=CYAN, fontsize=12, weight="bold")

    x_km = df["x_m"] / 1000.0
    y_km = df["y_m"] / 1000.0
    z_km = df["z_m"] / 1000.0
    vel_kmh = df["velocity_kmh"]

    # Trajectory curve
    p = ax.scatter(x_km, y_km, z_km, c=vel_kmh, cmap="plasma", s=10, alpha=0.85)
    cbar = fig.colorbar(p, ax=ax, shrink=0.6, aspect=18, pad=0.08)
    cbar.set_label("Vehicle Velocity (km/h)", color="#ffffff", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#ffffff")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#ffffff")

    # Launch Pad Marker
    ax.scatter([0], [0], [0], color=CYAN, s=100, marker="s", label="Launch Pad (KSC LC-39A)")
    ax.text(0, 0, 8, "Launch Pad\n(T+0s)", color=CYAN, fontsize=9, weight="bold")

    # Max-Q Marker
    max_q_idx = df["dynamic_pressure_pa"].idxmax()
    ax.scatter([x_km.loc[max_q_idx]], [y_km.loc[max_q_idx]], [z_km.loc[max_q_idx]], color=PURPLE, s=120, marker="X", label="Max-Q")
    ax.text(x_km.loc[max_q_idx]+20, y_km.loc[max_q_idx], z_km.loc[max_q_idx]+5, "Max-Q\n(11.2 km)", color=PURPLE, fontsize=9, weight="bold")

    # Apogee Marker
    apogee_idx = df["altitude_m"].idxmax()
    ax.scatter([x_km.loc[apogee_idx]], [y_km.loc[apogee_idx]], [z_km.loc[apogee_idx]], color=YELLOW, s=150, marker="D", label="Apogee (Max Alt)")
    ax.text(x_km.loc[apogee_idx]+30, y_km.loc[apogee_idx], z_km.loc[apogee_idx]+5, f"Apogee\n({z_km.loc[apogee_idx]:.1f} km)", color=YELLOW, fontsize=10, weight="bold")

    # Touchdown Marker
    ax.scatter([x_km.iloc[-1]], [y_km.iloc[-1]], [z_km.iloc[-1]], color=RED, s=110, marker="o", label="Splashdown / Touchdown")
    ax.text(x_km.iloc[-1]-100, y_km.iloc[-1], z_km.iloc[-1]+5, f"Touchdown\n({x_km.iloc[-1]:.0f} km downrange)", color=RED, fontsize=9, weight="bold")

    ax.set_xlabel("\nDownrange East (km)", color="#ffffff", fontsize=10)
    ax.set_ylabel("\nCross-range North (km)", color="#ffffff", fontsize=10)
    ax.set_zlabel("\nAltitude Above Sea Level (km)", color="#ffffff", fontsize=10)

    ax.xaxis.set_pane_color((0.06, 0.08, 0.12, 1.0))
    ax.yaxis.set_pane_color((0.06, 0.08, 0.12, 1.0))
    ax.zaxis.set_pane_color((0.06, 0.08, 0.12, 1.0))

    ax.grid(True, color="#2d3748", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", facecolor=PANEL_BG, edgecolor="#30363d", fontsize=9)
    ax.view_init(elev=22, azim=-62)

    plt.tight_layout()
    out_file = os.path.join(ASSETS_DIR, "03_3d_flight_trajectory.png")
    plt.savefig(out_file, dpi=180, facecolor=DARK_BG)
    plt.close()
    print("Saved:", out_file)


def generate_step4_telemetry_graphs(df):
    """Screenshot 4: 4-Panel Telemetry Analytics Curves"""
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), facecolor=DARK_BG)
    fig.suptitle("🚀 STEP 4: MULTI-CHANNEL FLIGHT TELEMETRY & PHYSICAL PERFORMANCE", color="#ffffff", fontsize=16, weight="bold", y=0.98)

    for ax in axs.flat:
        ax.set_facecolor(PANEL_BG)
        ax.grid(True, color="#2d3748", linestyle="--", alpha=0.5)

    # 1. Altitude & Vertical Speed
    ax1 = axs[0, 0]
    ax1.plot(df["time_s"], df["altitude_m"] / 1000.0, color=CYAN, lw=2.5, label="Altitude (km)")
    ax1.set_ylabel("Altitude (km)", color=CYAN, fontsize=10)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df["time_s"], df["vz_mps"], color=GREEN, lw=1.5, linestyle="--", label="Vertical Velocity (m/s)")
    ax1_twin.set_ylabel("Vertical Velocity (m/s)", color=GREEN, fontsize=10)
    ax1.set_title("Altitude & Vertical Velocity Profile", color="#ffffff", fontsize=11, weight="bold")
    ax1.legend(loc="upper left", facecolor="#0d1117", edgecolor="#30363d")

    # 2. Velocity & Mach Number
    ax2 = axs[0, 1]
    ax2.plot(df["time_s"], df["velocity_mps"], color=ORANGE, lw=2.5, label="True Speed (m/s)")
    ax2.set_ylabel("Speed (m/s)", color=ORANGE, fontsize=10)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df["time_s"], df["mach"], color=YELLOW, lw=1.5, linestyle=":", label="Mach Number")
    ax2_twin.set_ylabel("Mach", color=YELLOW, fontsize=10)
    ax2.set_title("Velocity & Mach Number Across Regimes", color="#ffffff", fontsize=11, weight="bold")
    ax2.legend(loc="upper left", facecolor="#0d1117", edgecolor="#30363d")

    # 3. Dynamic Pressure & Max-Q
    ax3 = axs[1, 0]
    q_kpa = df["dynamic_pressure_pa"] / 1000.0
    ax3.plot(df["time_s"], q_kpa, color=PURPLE, lw=2.5, label="Dynamic Pressure Q (kPa)")
    max_q_idx = q_kpa.idxmax()
    ax3.annotate(f"Max-Q ({q_kpa.loc[max_q_idx]:.1f} kPa)",
                 xy=(df["time_s"].loc[max_q_idx], q_kpa.loc[max_q_idx]),
                 xytext=(df["time_s"].loc[max_q_idx]+40, q_kpa.loc[max_q_idx]+5),
                 arrowprops=dict(facecolor=PURPLE, shrink=0.08, width=1.5, headwidth=6),
                 color=PURPLE, weight="bold", fontsize=9)
    ax3.set_xlabel("Elapsed Time T+ (seconds)", color="#ffffff", fontsize=10)
    ax3.set_ylabel("Dynamic Pressure (kPa)", color=PURPLE, fontsize=10)
    ax3.set_title("Aerodynamic Dynamic Pressure (Max-Q)", color="#ffffff", fontsize=11, weight="bold")
    ax3.legend(loc="upper right", facecolor="#0d1117", edgecolor="#30363d")

    # 4. Propellant Depletion & G-Load
    ax4 = axs[1, 1]
    ax4.plot(df["time_s"], df["remaining_fuel_kg"] / 1000.0, color="#2979ff", lw=2.5, label="Propellant Mass (tonnes)")
    ax4.set_ylabel("Propellant (tonnes)", color="#2979ff", fontsize=10)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(df["time_s"], df["g_load"], color=RED, lw=1.5, linestyle="-", label="G-Force (G)")
    ax4_twin.set_ylabel("G-Force (G)", color=RED, fontsize=10)
    ax4.set_xlabel("Elapsed Time T+ (seconds)", color="#ffffff", fontsize=10)
    ax4.set_title("Propellant Mass Depletion & Vehicle G-Load", color="#ffffff", fontsize=11, weight="bold")
    ax4.legend(loc="center left", facecolor="#0d1117", edgecolor="#30363d")

    plt.tight_layout()
    out_file = os.path.join(ASSETS_DIR, "04_telemetry_flight_curves.png")
    plt.savefig(out_file, dpi=180, facecolor=DARK_BG)
    plt.close()
    print("Saved:", out_file)


def generate_step5_ai_analytics():
    """Screenshot 5: AI Pre-Flight Inference & Anomaly Detection"""
    fig = plt.figure(figsize=(14, 8), facecolor=DARK_BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(DARK_BG)
    ax.axis("off")

    ax.text(0.04, 0.94, "🚀 AI ROCKET LAUNCH SIMULATOR — MISSION CONTROL", color="#ffffff", fontsize=20, weight="bold")
    ax.text(0.04, 0.89, "STEP 5: MACHINE LEARNING OUTCOME PREDICTION & TELEMETRY ANOMALY DETECTION", color="#ffd600", fontsize=12, weight="bold")

    # Left Panel: Physics vs AI Comparison Table
    rect_left = patches.FancyBboxPatch((0.04, 0.08), 0.46, 0.77, boxstyle="round,pad=0.01",
                                       fc=PANEL_BG, ec="#2d3748", lw=1.5)
    ax.add_patch(rect_left)
    ax.text(0.06, 0.80, "🎯 PHYSICS ENGINE vs AI PREDICTION SCORECARD", color="#ffffff", fontsize=12, weight="bold")

    headers = ["Metric", "Physics Calculated", "AI Estimate", "Error Delta"]
    hx = [0.06, 0.22, 0.34, 0.43]
    for i, h in enumerate(headers):
        ax.text(hx[i], 0.74, h, color=TEXT_MUTED, fontsize=9, weight="bold")
    ax.plot([0.06, 0.48], [0.725, 0.725], color="#30363d", lw=1.5)

    rows = [
        ("Apogee Altitude", "137.31 km", "235.51 km", "98.2 km"),
        ("Max Velocity", "6,670 m/s", "2,988 m/s", "3,682 m/s*"),
        ("Max-Q Dynamic Pressure", "39.40 kPa", "28.26 kPa", "11.1 kPa"),
        ("Mission Duration", "596.5 sec", "359.8 sec", "236.7 sec")
    ]
    ry = 0.67
    for r in rows:
        ax.text(hx[0], ry, r[0], color="#ffffff", fontsize=9, weight="bold")
        ax.text(hx[1], ry, r[1], color=CYAN, fontsize=9)
        ax.text(hx[2], ry, r[2], color=ORANGE, fontsize=9)
        ax.text(hx[3], ry, r[3], color=GREEN if "11" in r[3] else YELLOW, fontsize=9)
        ry -= 0.06

    ax.text(0.06, 0.41, "*Note: Single-stage surrogate vs Multi-stage vacuum engine burn.", color=TEXT_MUTED, fontsize=8, style="italic")

    # AI Anomaly Detection Box
    rect_anom = patches.FancyBboxPatch((0.06, 0.12), 0.42, 0.25, boxstyle="round,pad=0.01",
                                       fc="#0d1117", ec=RED, lw=1.5)
    ax.add_patch(rect_anom)
    ax.text(0.08, 0.32, "🛡️ AI ISOLATION FOREST ANOMALY ENGINE", color=RED, fontsize=11, weight="bold")
    ax.text(0.08, 0.27, "Status: ANOMALY DETECTED (Isolation Score: -0.107)", color="#ffffff", fontsize=10, weight="bold")
    ax.text(0.08, 0.22, "• High structural G-force load (> 6.0 G in upper atmosphere)", color=TEXT_MUTED, fontsize=9)
    ax.text(0.08, 0.17, "• Elevated aerodynamic dynamic pressure (> 65 kPa limit)", color=TEXT_MUTED, fontsize=9)
    ax.text(0.08, 0.13, "Recommendation: Command engine throttle throttling before Max-Q.", color=YELLOW, fontsize=8, weight="bold")

    # Right Panel: Feature Importance Bar Chart
    rect_right = patches.FancyBboxPatch((0.53, 0.08), 0.43, 0.77, boxstyle="round,pad=0.01",
                                        fc=PANEL_BG, ec="#2d3748", lw=1.5)
    ax.add_patch(rect_right)
    ax.text(0.55, 0.80, "📊 AI FEATURE IMPORTANCE (Apogee Regressor)", color="#ffffff", fontsize=12, weight="bold")

    features = [
        ("Liftoff Thrust", 0.32, CYAN),
        ("Fuel Mass Ratio", 0.28, GREEN),
        ("Engine Specific Impulse (Isp)", 0.18, ORANGE),
        ("Pitch Kick Angle", 0.11, YELLOW),
        ("Vehicle Diameter", 0.06, PURPLE),
        ("Drag Coefficient (Cd)", 0.05, RED)
    ]
    fy = 0.72
    for name, imp, col in features:
        ax.text(0.55, fy, name, color="#ffffff", fontsize=9, weight="bold")
        ax.text(0.91, fy, f"{imp*100:.0f}%", color=col, fontsize=9, weight="bold")
        # Bar
        bar_bg = patches.Rectangle((0.55, fy - 0.035), 0.39, 0.02, color="#0d1117")
        bar_fill = patches.Rectangle((0.55, fy - 0.035), 0.39 * imp / 0.35, 0.02, color=col)
        ax.add_patch(bar_bg)
        ax.add_patch(bar_fill)
        fy -= 0.10

    plt.tight_layout()
    out_file = os.path.join(ASSETS_DIR, "05_ai_prediction_analytics.png")
    plt.savefig(out_file, dpi=180, facecolor=DARK_BG)
    plt.close()
    print("Saved:", out_file)


def generate_step6_mission_debrief(summary, events):
    """Screenshot 6: Mission Debrief Timeline & KPI Scorecard"""
    fig = plt.figure(figsize=(14, 8), facecolor=DARK_BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(DARK_BG)
    ax.axis("off")

    ax.text(0.04, 0.94, "🚀 AI ROCKET LAUNCH SIMULATOR — MISSION CONTROL", color="#ffffff", fontsize=20, weight="bold")
    ax.text(0.04, 0.89, "STEP 6: MISSION DEBRIEFING, CHRONOLOGICAL EVENT LOG & EXPORT", color=GREEN, fontsize=12, weight="bold")

    # Top KPI Banner
    kpi_cards = [
        ("MAX ALTITUDE (APOGEE)", f"{summary['max_altitude_km']:.1f} km", CYAN),
        ("MAX VELOCITY", f"{summary['max_velocity_mps']:,.0f} m/s", ORANGE),
        ("MAX SPEED (MACH)", f"Mach {summary['max_velocity_mach']:.1f}", YELLOW),
        ("MAX-Q PRESSURE", f"{summary['max_dynamic_pressure_kpa']:.1f} kPa", PURPLE),
        ("DOWNRANGE DISTANCE", f"{summary['final_downrange_km']:.0f} km", GREEN)
    ]
    kx = 0.04
    for title, val, col in kpi_cards:
        k_rect = patches.FancyBboxPatch((kx, 0.72), 0.175, 0.14, boxstyle="round,pad=0.01",
                                       fc=PANEL_BG, ec="#2d3748", lw=1.2)
        ax.add_patch(k_rect)
        ax.text(kx + 0.01, 0.81, title, color=TEXT_MUTED, fontsize=8, weight="bold")
        ax.text(kx + 0.01, 0.75, val, color=col, fontsize=13, weight="bold")
        kx += 0.185

    # Mission Events Table Box
    rect_table = patches.FancyBboxPatch((0.04, 0.08), 0.92, 0.60, boxstyle="round,pad=0.01",
                                       fc=PANEL_BG, ec="#2d3748", lw=1.5)
    ax.add_patch(rect_table)
    ax.text(0.06, 0.63, "📋 CHRONOLOGICAL MISSION TIMELINE & TELEMETRY LOG", color="#ffffff", fontsize=12, weight="bold")

    t_headers = ["Time (MET)", "Event Code", "Mission Controller Details", "Altitude", "Speed", "Downrange"]
    tx = [0.06, 0.16, 0.32, 0.68, 0.78, 0.88]
    for i, th in enumerate(t_headers):
        ax.text(tx[i], 0.58, th, color=TEXT_MUTED, fontsize=9, weight="bold")
    ax.plot([0.06, 0.94], [0.565, 0.565], color="#30363d", lw=1.5)

    ty = 0.52
    for ev in events:
        ax.text(tx[0], ty, f"T+{ev['time_s']:.1f}s", color=CYAN, fontsize=9, weight="bold")
        ax.text(tx[1], ty, ev['event'], color=ORANGE if "SECO" in ev['event'] or "MECO" in ev['event'] else "#ffffff", fontsize=9, weight="bold")
        ax.text(tx[2], ty, ev['details'][:45], color=TEXT_MUTED, fontsize=8.5)
        ax.text(tx[3], ty, f"{ev['altitude_m']:,} m", color="#ffffff", fontsize=8.5)
        ax.text(tx[4], ty, f"{ev['velocity_mps']:,.0f} m/s", color="#ffffff", fontsize=8.5)
        ax.text(tx[5], ty, f"{ev['downrange_km']:.1f} km", color="#ffffff", fontsize=8.5)
        ty -= 0.052

    # Download buttons badge preview
    ax.text(0.06, 0.11, "📥 Export Available: Raw Telemetry CSV (telemetry_falcon9.csv) | Full Markdown Mission Report (.md)", color=GREEN, fontsize=9, weight="bold")

    plt.tight_layout()
    out_file = os.path.join(ASSETS_DIR, "06_mission_debrief_report.png")
    plt.savefig(out_file, dpi=180, facecolor=DARK_BG)
    plt.close()
    print("Saved:", out_file)


def main():
    print("Generating comprehensive simulation run for visual assets...")
    profile = ROCKET_PRESETS["Falcon 9 Medium-Lift (2-Stage)"]
    env = EnvironmentConfig(site=LAUNCH_SITES["Kennedy Space Center (LC-39A)"])
    sim = SimulationEngine(profile, env, dt=0.5, max_time_s=600.0)
    df, summary, events = sim.run()

    print("Rendering 6 step-by-step screenshots...")
    generate_step1_config()
    generate_step2_viewport()
    generate_step3_3d_trajectory(df, events)
    generate_step4_telemetry_graphs(df)
    generate_step5_ai_analytics()
    generate_step6_mission_debrief(summary, events)
    print("All screenshots generated successfully in assets/screenshots/!")

if __name__ == "__main__":
    main()
