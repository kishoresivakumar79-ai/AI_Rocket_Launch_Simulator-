# AI Model Evaluation, Feature Importance & Residual Analysis
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any

def create_feature_importance_figure(model, feature_names: list, target_name: str, theme: str = "plotly_dark") -> go.Figure:
    """Plots top feature importances for a trained Random Forest model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)

    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig = go.Figure(go.Bar(
        x=sorted_importances,
        y=sorted_features,
        orientation='h',
        marker=dict(color="#00e5ff")
    ))

    fig.update_layout(
        title=f"🧠 AI Feature Importance: {target_name.replace('_', ' ').title()}",
        template=theme,
        xaxis=dict(title="Relative Importance Score", gridcolor="#333"),
        yaxis=dict(gridcolor="#333"),
        margin=dict(l=120, r=20, t=50, b=40)
    )
    return fig


def create_actual_vs_predicted_figure(y_actual: list, y_predicted: list, target_name: str, unit: str = "km", theme: str = "plotly_dark") -> go.Figure:
    """Plots Predicted vs Actual scatter with ideal 1:1 identity line."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_actual,
        y=y_predicted,
        mode="markers",
        marker=dict(color="#ffd600", size=7, opacity=0.8),
        name="Test Simulations"
    ))

    # 1:1 line
    min_val = min(min(y_actual), min(y_predicted))
    max_val = max(max(y_actual), max(y_predicted))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Ideal 1:1 Prediction",
        line=dict(color="#ff1744", dash="dash", width=2)
    ))

    fig.update_layout(
        title=f"AI Regression Accuracy: {target_name.replace('_', ' ').title()}",
        template=theme,
        xaxis=dict(title=f"Actual Physics Output ({unit})", gridcolor="#333"),
        yaxis=dict(title=f"AI Predicted Output ({unit})", gridcolor="#333"),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

