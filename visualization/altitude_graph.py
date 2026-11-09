# Altitude Telemetry Visualization
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd

def create_altitude_figure(df: pd.DataFrame, theme: str = "plotly_dark") -> go.Figure:
    """Creates a high-precision Plotly altitude & vertical speed telemetry chart."""
    fig = go.Figure()

    # Altitude trace
    fig.add_trace(go.Scatter(
        x=df["time_s"],
        y=df["altitude_m"] / 1000.0,
        mode="lines",
        name="Altitude (km)",
        line=dict(color="#00e5ff", width=3),
        hovertemplate="Time: %{x:.1f} s<br>Altitude: %{y:.2f} km<extra></extra>"
    ))

    # Vertical Velocity trace
    fig.add_trace(go.Scatter(
        x=df["time_s"],
        y=df["vz_mps"],
        mode="lines",
        name="Vertical Velocity (m/s)",
        line=dict(color="#00e676", width=2, dash="dash"),
        yaxis="y2",
        hovertemplate="Time: %{x:.1f} s<br>V_z: %{y:.1f} m/s<extra></extra>"
    ))

    # Apogee callout
    apogee_idx = df["altitude_m"].idxmax()
    apogee_t = df.loc[apogee_idx, "time_s"]
    apogee_h = df.loc[apogee_idx, "altitude_m"] / 1000.0

    fig.add_annotation(
        x=apogee_t,
        y=apogee_h,
        text=f"Apogee: {apogee_h:.1f} km",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#ffeb3b",
        font=dict(color="#ffeb3b", size=12),
        bgcolor="rgba(20,20,30,0.85)",
        bordercolor="#ffeb3b"
    )

    fig.update_layout(
        title="🚀 Altitude & Vertical Velocity Telemetry",
        template=theme,
        xaxis=dict(title="Mission Elapsed Time (seconds)", gridcolor="#333"),
        yaxis=dict(title="Altitude (km)", gridcolor="#333", side="left"),
        yaxis2=dict(title="Vertical Velocity (m/s)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="x unified"
    )
    return fig


def plot_altitude_matplotlib(df: pd.DataFrame, show: bool = True):
    """Matplotlib equivalent matching Section 9 prototype."""
    plt.figure(figsize=(10, 5))
    plt.plot(df["time_s"], df["altitude_m"], color="blue", label="Altitude (m)")
    plt.title("Rocket Altitude vs Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Altitude (meters)")
    plt.grid(True)
    plt.legend()
    if show:
        plt.show()

