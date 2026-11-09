# 3D Interactive & Animated Rocket Trajectory Visualization
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Dict, Any, List

def create_3d_trajectory_figure(
    df: pd.DataFrame,
    events: Optional[list] = None,
    theme: str = "plotly_dark",
    animated: bool = True,
    max_frames: int = 50
) -> go.Figure:
    """
    Constructs an interactive 3D flight trajectory in Plotly.
    If animated=True, includes built-in Play/Pause buttons, a timeline scrubber,
    and an animated rocket vehicle traveling along the trajectory.
    """
    fig = go.Figure()

    # Convert coordinates to kilometers for intuitive viewing
    x_km = df["x_m"] / 1000.0
    y_km = df["y_m"] / 1000.0
    z_km = df["z_m"] / 1000.0
    vel_kmh = df["velocity_kmh"]
    time_s = df["time_s"]

    # 1. Full trajectory guide path (faint dotted path showing planned trajectory)
    fig.add_trace(go.Scatter3d(
        x=x_km,
        y=y_km,
        z=z_km,
        mode="lines",
        name="Planned Flight Path",
        line=dict(color="rgba(0, 229, 255, 0.3)", width=3, dash="dot"),
        hoverinfo="skip"
    ))

    # 2. Dynamic Trajectory Trail (grows during animation)
    fig.add_trace(go.Scatter3d(
        x=[x_km.iloc[0]],
        y=[y_km.iloc[0]],
        z=[z_km.iloc[0]],
        mode="lines",
        name="Traveled Flight Path",
        line=dict(
            color=vel_kmh.iloc[:1],
            colorscale="Viridis",
            width=7,
            cmin=0,
            cmax=max(100.0, float(vel_kmh.max())),
            colorbar=dict(title="Speed (km/h)", x=1.05, len=0.6)
        )
    ))

    # 3. Animated Rocket Vehicle Position Marker (glowing icon / diamond)
    fig.add_trace(go.Scatter3d(
        x=[x_km.iloc[0]],
        y=[y_km.iloc[0]],
        z=[z_km.iloc[0]],
        mode="markers+text",
        name="Rocket Vehicle (Active)",
        marker=dict(
            size=10,
            color="#ff1744",
            symbol="diamond",
            line=dict(color="#ffffff", width=2)
        ),
        text=["🚀 Rocket (T+0s)"],
        textposition="top center",
        hovertemplate=(
            "<b>🚀 %{text}</b><br>"
            "Altitude: %{z:.2f} km<br>"
            "Downrange: %{x:.2f} km<extra></extra>"
        )
    ))

    # 4. Launch Pad Ground marker
    fig.add_trace(go.Scatter3d(
        x=[x_km.iloc[0]],
        y=[y_km.iloc[0]],
        z=[z_km.iloc[0]],
        mode="markers+text",
        name="Launch Pad",
        marker=dict(size=8, color="#00e5ff", symbol="square"),
        text=["Launch Pad"],
        textposition="top center"
    ))

    # 5. Apogee Marker
    apogee_idx = df["altitude_m"].idxmax()
    fig.add_trace(go.Scatter3d(
        x=[x_km.loc[apogee_idx]],
        y=[y_km.loc[apogee_idx]],
        z=[z_km.loc[apogee_idx]],
        mode="markers+text",
        name="Apogee (Max Alt)",
        marker=dict(size=9, color="#ffd600", symbol="diamond"),
        text=[f"Apogee: {z_km.loc[apogee_idx]:.1f} km"],
        textposition="top center"
    ))

    # 6. Touchdown / Landing marker
    fig.add_trace(go.Scatter3d(
        x=[x_km.iloc[-1]],
        y=[y_km.iloc[-1]],
        z=[z_km.iloc[-1]],
        mode="markers+text",
        name="Touchdown / Final Position",
        marker=dict(size=8, color="#ff1744", symbol="circle"),
        text=[f"Touchdown ({x_km.iloc[-1]:.1f} km)"],
        textposition="bottom center"
    ))

    # 7. Staging and Max-Q event markers
    if events:
        for ev in events:
            ev_name = ev.get("event", "")
            if ev_name in ["STAGE SEPARATION", "MAX-Q", "MECO"]:
                t_ev = ev.get("time_s", 0.0)
                closest_idx = (df["time_s"] - t_ev).abs().idxmin()
                marker_color = "#e040fb" if ev_name == "MAX-Q" else "#ff9100"
                fig.add_trace(go.Scatter3d(
                    x=[x_km.loc[closest_idx]],
                    y=[y_km.loc[closest_idx]],
                    z=[z_km.loc[closest_idx]],
                    mode="markers+text",
                    name=ev_name,
                    marker=dict(size=7, color=marker_color, symbol="cross"),
                    text=[f"{ev_name} (T+{t_ev:.0f}s)"],
                    textposition="top center",
                    showlegend=False
                ))

    # Build Animation Frames if requested
    if animated and len(df) > 1:
        step = max(1, len(df) // max_frames)
        sample_indices = list(range(0, len(df), step))
        if sample_indices[-1] != len(df) - 1:
            sample_indices.append(len(df) - 1)

        frames = []
        slider_steps = []

        for idx in sample_indices:
            t_curr = time_s.iloc[idx]
            sub_x = x_km.iloc[:idx+1]
            sub_y = y_km.iloc[:idx+1]
            sub_z = z_km.iloc[:idx+1]
            sub_v = vel_kmh.iloc[:idx+1]
            phase_curr = df["phase"].iloc[idx]

            frame_data = [
                # Trace 0 (planned path) remains static
                go.Scatter3d(x=x_km, y=y_km, z=z_km),
                # Trace 1: Traveled trail up to current point
                go.Scatter3d(
                    x=sub_x,
                    y=sub_y,
                    z=sub_z,
                    line=dict(color=sub_v, colorscale="Viridis", width=7)
                ),
                # Trace 2: Active Rocket Marker
                go.Scatter3d(
                    x=[x_km.iloc[idx]],
                    y=[y_km.iloc[idx]],
                    z=[z_km.iloc[idx]],
                    text=[f"🚀 Rocket T+{t_curr:.0f}s ({phase_curr})"]
                )
            ]

            frame_name = f"T+{t_curr:.0f}s"
            frames.append(go.Frame(data=frame_data, name=frame_name))

            slider_steps.append({
                "args": [
                    [frame_name],
                    {"frame": {"duration": 100, "redraw": True}, "mode": "immediate"}
                ],
                "label": f"{t_curr:.0f}s",
                "method": "animate"
            })

        fig.frames = frames

        # Add Play / Pause Controls and Timeline Slider
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=0.05,
                    y=0.05,
                    xanchor="left",
                    yanchor="bottom",
                    bgcolor="#1e222d",
                    bordercolor="#00e5ff",
                    font=dict(color="#ffffff"),
                    buttons=[
                        dict(
                            label="▶ PLAY 3D FLIGHT",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 80, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0}
                                }
                            ]
                        ),
                        dict(
                            label="⏸ PAUSE",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}
                                }
                            ]
                        )
                    ]
                )
            ],
            sliders=[
                dict(
                    active=0,
                    yanchor="bottom",
                    xanchor="left",
                    currentvalue=dict(
                        font=dict(size=14, color="#00e5ff"),
                        prefix="Mission Time: ",
                        visible=True,
                        xanchor="right"
                    ),
                    transition=dict(duration=50),
                    pad=dict(b=10, t=10),
                    len=0.85,
                    x=0.1,
                    y=0.0,
                    steps=slider_steps,
                    font=dict(color="#ffffff"),
                    bgcolor="#1e222d",
                    bordercolor="#333"
                )
            ]
        )

    # Configure 3D camera and aerospace grid layout
    fig.update_layout(
        title="🛰️ Interactive 3D Animated Rocket Trajectory (Click ▶ PLAY to watch vehicle flight)",
        template=theme,
        margin=dict(l=10, r=10, t=40, b=10),
        scene=dict(
            xaxis=dict(title="Downrange X (km)", gridcolor="#333", backgroundcolor="#111"),
            yaxis=dict(title="Cross-range Y (km)", gridcolor="#333", backgroundcolor="#111"),
            zaxis=dict(title="Altitude Z (km)", gridcolor="#333", backgroundcolor="#111"),
            aspectmode="data",
            camera=dict(
                eye=dict(x=-1.5, y=-1.5, z=1.2)
            )
        )
    )
    return fig


def show_trajectory(data: np.ndarray):
    """Prototype convenience function."""
    time = data[:, 0]
    altitude = data[:, 1]
    x = np.zeros(len(time))
    y = np.zeros(len(time))
    z = altitude

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines",
        name="Rocket Trajectory",
        line=dict(color="#00e5ff", width=4)
    ))
    fig.update_layout(
        title="3D Rocket Flight Trajectory",
        scene=dict(
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            zaxis_title="Altitude (m)"
        )
    )
    fig.show()
