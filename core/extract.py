import cv2
import numpy as np
from typing import Optional


def calculate_sharpness(image: np.ndarray) -> float:
    """
    Evaluates the sharpness of a frame using the Laplacian variance.
    Higher values generally indicate a sharper, more 'in-focus' image.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # The Laplacian highlights regions of rapid intensity change (edges)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract_frame_at_time(cap: cv2.VideoCapture, timestamp: float) -> Optional[np.ndarray]:
    """
    Extracts a single frame from a VideoCapture object at a specific time.

    Args:
        cap: An open cv2.VideoCapture object.
        timestamp: The time in seconds where the frame should be extracted.

    Returns:
        The frame as a numpy array, or None if the timestamp is out of bounds.
    """
    # Convert seconds to milliseconds for OpenCV
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)

    success, frame = cap.read()
    if success:
        return frame
    return None
