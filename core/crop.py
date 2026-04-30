import cv2
import numpy as np
from typing import Dict


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders coordinates: [top-left, top-right, bottom-right, bottom-left].
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Top-left has the smallest sum, bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has the smallest difference, bottom-left has the largest
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def passed_quads(state: Dict) -> Dict:
    """
    Performs a perspective transform on passed_indices quads to extract de-skewed crops 
    respecting their original aspect ratio and point order.
    """
    image = state.get("raw_frame")
    quads = state.get("quads", [])
    passed_indices = state.get("passed_indices", list(range(len(quads))))

    if image is None or not quads:
        return {"crops": []}

    extracted_crops = []
    for idx in passed_indices:
        # 1. Properly order the 4 corners
        raw_pts = quads[idx].reshape(4, 2).astype(np.float32)
        pts = order_points(raw_pts)
        (tl, tr, br, bl) = pts

        # 2. Calculate the width and height of the new image
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_w = max(int(width_a), int(width_b))

        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_h = max(int(height_a), int(height_b))

        # 3. Define destination points for the de-skewed "bird's eye view"
        dst_pts = np.array([
            [0, 0],
            [max_w - 1, 0],
            [max_w - 1, max_h - 1],
            [0, max_h - 1]
        ], dtype="float32")

        # 4. Compute the transform matrix and warp
        matrix = cv2.getPerspectiveTransform(pts, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (max_w, max_h))

        extracted_crops.append(warped)

    return {"crops": extracted_crops}
