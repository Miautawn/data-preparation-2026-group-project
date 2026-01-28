from typing import Sequence

import plotly.graph_objects as go


def plot_target_vs_predicted_heartrate(
    heart_rate: Sequence[float], heart_rate_predicted: Sequence[float], title: str = ""
):
    fig = go.Figure()
    x = list(range(len(heart_rate)))

    fig.add_trace(go.Scatter(y=heart_rate, x=x, mode="lines", name="target"))
    fig.add_trace(
        go.Scatter(y=heart_rate_predicted, x=x, mode="lines", name="prediction")
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time Step",
        yaxis_title="Heart Rate (bpm)",
    )

    return fig
