import cv2
import numpy as np
from typing import Dict
from . import visualize as viz


def find_quads(state: Dict, min_area: int = 1000, epsilon: float = 0.02) -> Dict:
    """
    Identifies 4-sided polygons and labels them with a unique ID index.

    Args:
        state: Dictionary containing 'current_image' (binary) and 'raw_frame'.
        min_area: Minimum pixel area to consider a shape.
        epsilon: Corner approximation sensitivity.

    Returns:
        Updated state with 'quads' list and 'current_image' showing labeled boxes.
    """
    binary_image = state["current_image"]
    raw_frame = state.get("raw_frame", binary_image)

    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    quads_list = []
    labels_list = []

    # 1. Detection Logic
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon * peri, True)

            if len(approx) == 4:
                quads_list.append(approx)
                # Use the 0-based index as the label
                labels_list.append(str(len(quads_list) - 1))

    # 2. Prepare the data for the visualization function
    quad_group: viz.QuadGroup = {
        "quads": quads_list,
        "labels": labels_list,
        "color": (0, 255, 0)  # Green text and borders
    }

    # 3. Use the visualize function to create the debug image
    debug_img = viz.draw_multiple_quad_groups(raw_frame, [quad_group])

    return {
        "quads": quads_list,
        "current_image": debug_img,
        "auto_save": debug_img
    }
