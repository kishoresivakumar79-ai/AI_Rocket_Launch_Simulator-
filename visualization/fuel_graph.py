# Fuel, Mass, Thrust & G-Force Telemetry Visualization
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd

def create_fuel_mass_figure(df: pd.DataFrame, theme: str = "plotly_dark") -> go.Figure:
    """Creates a chart for fuel consumption, total mass, thrust, and acceleration G-force."""
    fig = go.Figure()

    # Remaining Fuel trace
    fig.add_trace(go.Scatter(
        x=df["time_s"],
        y=df["remaining_fuel_kg"],
        mode="lines",
        name="Propellant Mass (kg)",
        line=dict(color="#2979ff", width=2.5),
        hovertemplate="Time: %{x:.1f} s<br>Fuel: %{y:,.0f} kg<extra></extra>"
    ))

    # Total Mass trace
    fig.add_trace(go.Scatter(
        x=df["time_s"],
        y=df["mass_kg"],
        mode="lines",
        name="Total Vehicle Mass (kg)",
        line=dict(color="#b0bec5", width=2, dash="dash"),
        hovertemplate="Time: %{x:.1f} s<br>Vehicle Mass: %{y:,.0f} kg<extra></extra>"
    ))

    # G-Force trace
    fig.add_trace(go.Scatter(
        x=df["time_s"],
        y=df["g_load"],
        mode="lines",
        name="Acceleration (G-load)",
        line=dict(color="#ff1744", width=2),
        yaxis="y2",
        hovertemplate="Time: %{x:.1f} s<br>G-Load: %{y:.2f} G<extra></extra>"
    ))

    fig.update_layout(
        title="⛽ Propellant Consumption & Acceleration (G-Load)",
        template=theme,
        xaxis=dict(title="Mission Elapsed Time (seconds)", gridcolor="#333"),
        yaxis=dict(title="Mass (kg)", gridcolor="#333", side="left"),
        yaxis2=dict(title="G-Force (G)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="x unified"
    )
    return fig


def plot_fuel_matplotlib(df: pd.DataFrame, show: bool = True):
    """Matplotlib equivalent."""
    plt.figure(figsize=(10, 5))
    plt.plot(df["time_s"], df["remaining_fuel_kg"], color="purple", label="Fuel (kg)")
    plt.title("Rocket Fuel Mass vs Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Fuel Mass (kg)")
    plt.grid(True)
    plt.legend()
    if show:
        plt.show()

