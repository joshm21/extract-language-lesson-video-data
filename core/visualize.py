import cv2
import numpy as np
from typing import List, Tuple, TypedDict


class QuadGroup(TypedDict):
    quads: List[np.ndarray]
    labels: List[str]
    color: Tuple[int, int, int]


def draw_multiple_quad_groups(
    image: np.ndarray,
    groups: List[QuadGroup]
) -> np.ndarray:
    """
    Draws multiple groups of contours and labels on an image.

    Args:
        image: The canvas to draw on.
        groups: A list of dictionaries, each containing:
                - 'quads': List of 4-point polygons.
                - 'labels': List of strings corresponding to each quad.
                - 'color': BGR tuple for that group.
    """
    canvas = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2

    for group in groups:
        quads = group['quads']
        labels = group['labels']
        color = group['color']

        for quad, label in zip(quads, labels):
            # 1. Draw the polygon border
            cv2.drawContours(canvas, [quad], -1, color, 3)

            # 2. Calculate Centered Text Position
            M = cv2.moments(quad)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                (t_w, t_h), _ = cv2.getTextSize(label, font, scale, thickness)

                # Center the text[cite: 1]
                text_x = cX - (t_w // 2)
                text_y = cY + (t_h // 2)

                # 3. Draw Shadow (Black) then Foreground (Color)[cite: 1]
                cv2.putText(canvas, str(label), (text_x + 1, text_y + 1),
                            font, scale, (0, 0, 0), thickness + 1)
                cv2.putText(canvas, str(label), (text_x, text_y),
                            font, scale, color, thickness)

    return canvas
