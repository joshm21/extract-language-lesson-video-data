import cv2
import numpy as np


def get_perspective_transform(image: np.ndarray, pts: np.ndarray, padding: int = 0) -> np.ndarray:
    """
    Warps a quadrilateral section of an image into a straight, bird's-eye view.

    Args:
        image: The source frame.
        pts: The 4 corner points.
        padding: Optional extra pixels to crop inward (to remove border artifacts).
    """
    # 1. Transform the raw native quad into ordered float32 points
    ordered_pts = order_points(pts)

    (tl, tr, br, bl) = ordered_pts

    # Calculate dimensions
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # 2. Calculate the transform matrix and warp
    # cv2.getPerspectiveTransform needs float32 inputs
    matrix = cv2.getPerspectiveTransform(ordered_pts, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    if padding > 0:
        warped = warped[padding:-padding, padding:-padding]

    return warped


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def all(frame, quads):
    return [get_perspective_transform(frame, q) for q in quads]
