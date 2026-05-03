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

    # 1. Detection Logic: Collect all quads first without labeling
    found_quads = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon * peri, True)
            if len(approx) == 4:
                found_quads.append(approx)

    # 2. Sorting Logic: Top-to-Bottom, then Left-to-Right
    # We use the top-left (y, x) of the bounding box as the sorting key
    def get_sort_key(q):
        x, y, w, h = cv2.boundingRect(q)
        # Trick: Rounding y helps group cards into the same 'row' if the table is slightly tilted
        return (y // 100, x)
        # return (y, x)

    quads_list = sorted(found_quads, key=get_sort_key)

    # 3. Labeling Logic: Assign IDs based on the NEW sorted order
    labels_list = [str(i) for i in range(len(quads_list))]

    # 4. Prepare data for visualization[cite: 4]
    quad_group: viz.QuadGroup = {
        "quads": quads_list,
        "labels": labels_list,
        "color": (0, 255, 0)
    }

    debug_img = viz.draw_multiple_quad_groups(raw_frame, [quad_group])

    return {
        "quads": quads_list,
        "current_image": debug_img,
        "auto_save": debug_img
    }
