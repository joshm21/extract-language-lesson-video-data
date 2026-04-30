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
        epsilon: Higher = more 'rounded' corners allowed.
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
                quads.append(approx)
    return quads
