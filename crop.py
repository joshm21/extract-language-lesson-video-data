import cv2
import numpy as np


def get_perspective_transform(image: np.ndarray, pts: np.ndarray, padding: int = 0) -> np.ndarray:
    """
    Warps a quadrilateral section of an image into a straight, bird's-eye view.

    Args:
        image: The source frame.
        pts: The 4 corner points (should be ordered via detect.order_points).
        padding: Optional extra pixels to crop inward (to remove border artifacts).
    """
    # 1. Determine dimensions of the new image
    (tl, tr, br, bl) = pts

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # 2. Define destination points (a perfect rectangle)
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # 3. Calculate the transform matrix and warp
    matrix = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    # 4. Apply optional padding
    if padding > 0:
        warped = warped[padding:-padding, padding:-padding]

    return warped
