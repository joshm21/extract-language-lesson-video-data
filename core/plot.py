import matplotlib.pyplot as plt
import numpy as np
import io
import cv2
from typing import Dict, Any, List
from .score import props

import logging
# Silence the technical 'STREAM' chatter from image backends
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)


def feature_space(
    state: Dict[str, Any],
    x_axis: str = props.area,
    y_axis: str = props.edge_density,
    title: str = "Feature Distribution"
) -> Dict[str, Any]:
    """
    Plots all detected quads in a 2D feature space. 

    This helps students see the 'clusters' in the data visually before 
    applying ML algorithms like K-Means or KNN.

    Args:
        state: Pipeline state containing 'scores'.
        x_axis: Metric for the X-axis (from score.props).
        y_axis: Metric for the Y-axis (from score.props).
        title: The heading for the plot.
    """
    scores = state.get("scores", [])

    if not scores:
        return {}

    # 1. Extract raw feature data from the scores list
    x_vals = [s.get(x_axis, 0.0) for s in scores]
    y_vals = [s.get(y_axis, 0.0) for s in scores]
    indices = list(range(len(scores)))

    # 2. Create Plot
    fig, ax = plt.subplots(figsize=(7, 5))

    # Use a neutral color for all points
    ax.scatter(x_vals, y_vals, c='#3498db', s=60,
               edgecolors='white', alpha=0.8)

    # 3. Annotate with Quad Index
    # This matches the IDs seen in the detection and clustering images
    for i, idx in enumerate(indices):
        ax.annotate(f" {idx}", (x_vals[i], y_vals[i]),
                    fontsize=10, color="#2c3e50")

    # 4. Labels and Styling
    ax.set_xlabel(x_axis.replace('_', ' ').title())
    ax.set_ylabel(y_axis.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.5)

    # 5. Convert to OpenCV image for the Runner's auto_save logic
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)

    file_bytes = np.asarray(bytearray(buf.read()), dtype=np.uint8)
    plot_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Return as auto_save so the Runner writes the .jpg
    return {
        "auto_save": plot_img
    }
