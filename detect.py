import cv2
import numpy as np
from typing import List


def get_quads(
    binary_image: np.ndarray,
    min_area: int = 1000,
    epsilon: float = 0.02
) -> List[np.ndarray]:
    """
    Extracts and orders 4-sided polygons from a binary mask.

    Args:
        binary_image: The black-and-white input from a segmenter.
        min_area: Higher = stricter size filter.
        epsilon: Higher = more 'rounded' corners allowed[cite: 1].
    """
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    quads = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon * peri, True)

            if len(approx) == 4:
                # We order the points here so the results are ready for cropping
                ordered = order_points(approx)
                quads.append(ordered)
    return quads


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders coordinates consistently: [top-left, top-right, bottom-right, bottom-left].
    Essential for consistent cropping and deskewing.
    """
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
