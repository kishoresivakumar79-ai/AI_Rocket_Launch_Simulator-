# Velocity, Mach & Dynamic Pressure Telemetry Visualization
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd

def create_velocity_figure(df: pd.DataFrame, theme: str = "plotly_dark") -> go.Figure:
    """Creates a high-precision Plotly velocity, Mach and Max-Q chart."""
    fig = go.Figure()

    # Total Speed trace
    fig.add_trace(go.Scatter(
        x=df["time_s"],
        y=df["velocity_mps"],
        mode="lines",
        name="Velocity (m/s)",
        line=dict(color="#ff9100", width=3),
        hovertemplate="Time: %{x:.1f} s<br>Speed: %{y:.1f} m/s<extra></extra>"
    ))

    # Mach Number trace
    fig.add_trace(go.Scatter(
        x=df["time_s"],
        y=df["mach"],
        mode="lines",
        name="Mach Number",
        line=dict(color="#ffd600", width=2, dash="dot"),
        yaxis="y2",
        hovertemplate="Time: %{x:.1f} s<br>Mach: %{y:.2f}<extra></extra>"
    ))

    # Dynamic Pressure Q trace (converted to kPa)
    fig.add_trace(go.Scatter(
        x=df["time_s"],
        y=df["dynamic_pressure_pa"] / 1000.0,
        mode="lines",
        name="Dynamic Pressure Q (kPa)",
        line=dict(color="#e040fb", width=2, dash="dash"),
        yaxis="y3",
        hovertemplate="Time: %{x:.1f} s<br>Q: %{y:.1f} kPa<extra></extra>"
    ))

    # Max-Q callout
    max_q_idx = df["dynamic_pressure_pa"].idxmax()
    max_q_t = df.loc[max_q_idx, "time_s"]
    max_q_v = df.loc[max_q_idx, "velocity_mps"]
    max_q_val = df.loc[max_q_idx, "dynamic_pressure_pa"] / 1000.0

    if max_q_val > 5.0:
        fig.add_annotation(
            x=max_q_t,
            y=max_q_v,
            text=f"Max-Q ({max_q_val:.1f} kPa)",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#e040fb",
            font=dict(color="#e040fb", size=11),
            bgcolor="rgba(20,20,30,0.85)",
            bordercolor="#e040fb"
        )

    fig.update_layout(
        title="⚡ Velocity, Mach & Dynamic Pressure Telemetry",
        template=theme,
        xaxis=dict(title="Mission Elapsed Time (seconds)", gridcolor="#333"),
        yaxis=dict(title="Velocity (m/s)", gridcolor="#333", side="left"),
        yaxis2=dict(title="Mach", overlaying="y", side="right", showgrid=False, range=[0, max(6.0, df['mach'].max() * 1.1)]),
        yaxis3=dict(title="Dynamic Pressure (kPa)", overlaying="y", side="right", anchor="free", position=0.96, showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=50, t=50, b=40),
        hovermode="x unified"
    )
    return fig


def plot_velocity_matplotlib(df: pd.DataFrame, show: bool = True):
    """Matplotlib equivalent."""
    plt.figure(figsize=(10, 5))
    plt.plot(df["time_s"], df["velocity_mps"], color="red", label="Velocity (m/s)")
    plt.title("Rocket Velocity vs Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Velocity (m/s)")
    plt.grid(True)
    plt.legend()
    if show:
        plt.show()

