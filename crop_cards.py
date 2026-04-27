import cv2
import numpy as np


def deskew_and_crop(image: np.ndarray, pts: list) -> np.ndarray:
    """
    Applies perspective transform to extract a bird's-eye view of a card.

    Args:
        image: Source image array.
        pts: List of 4 coordinate points [[x,y], ...].

    Returns:
        The warped and cropped card image.
    """
    rect = np.array(pts, dtype="float32")
    (tl, tr, br, bl) = rect

    # Calculate max width and height for the new image
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

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))
