import cv2
import numpy as np
from typing import Optional, Dict


def calculate_sharpness(state: Dict) -> Dict:
    """
    Evaluates the sharpness of a frame using the Laplacian variance.
    Higher values generally indicate a sharper, more 'in-focus' image.
    """
    # Pull image from state; assume we use 'raw_frame' if available
    image = state.get("raw_frame")
    if image is None:
        return {}

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # The Laplacian highlights regions of rapid intensity change (edges)
    sharpness_val = cv2.Laplacian(gray, cv2.CV_64F).var()

    return {"sharpness": sharpness_val}


def at_current_timestamp(state: Dict) -> Dict:
    """
    Extracts a single frame from a VideoCapture object at a specific time.

    Args:
        cap: An open cv2.VideoCapture object.
        timestamp: The time in seconds where the frame should be extracted.

    Returns:
        The frame as a numpy array, or None if the timestamp is out of bounds.
    """
    # Pull required context from state
    data_dir = state["data_dir"]
    video_id = state["video_id"]
    timestamp = state["timestamp"]

    video_path = data_dir / video_id / "video.mp4"
    cap = cv2.VideoCapture(str(video_path))

    # Convert seconds to milliseconds for OpenCV
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)

    success, frame = cap.read()
    cap.release()  # Ensure resource is freed

    if success:
        # Trigger auto_save for the workshop
        return {"raw_frame": frame, "current_image": frame, "auto_save": frame}

    return {}
