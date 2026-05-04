import cv2
import numpy as np
from typing import Optional, Dict


import cv2
import numpy as np
from typing import Dict


def get_sharpness_score(frame: np.ndarray) -> float:
    """
    Stand-alone helper to evaluate frame sharpness.
    Uses Laplacian variance: higher = sharper.
    """
    if frame is None:
        return 0.0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def at_sharpest_in_window(state: Dict, window_seconds: float = 1.0) -> Dict:
    """
    Finds the sharpest frame within a +/- window of the target timestamp.
    Handles edge cases to prevent seeking outside video bounds.
    """
    data_dir = state["data_dir"]
    video_id = state["video_id"]
    target_ts = state["timestamp"]

    video_path = data_dir / video_id / "video.mp4"
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {}

    # 1. Calculate Bounds & Metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps

    start_time = max(0, target_ts - window_seconds)
    end_time = min(duration, target_ts + window_seconds)

    # 2. Iterate and Score
    # Sampling 10 times per second (0.1s steps) is a good performance/quality trade-off
    step = 0.1
    curr_time = start_time

    best_frame = None
    best_score = -1.0
    best_ts = target_ts

    while curr_time <= end_time:
        cap.set(cv2.CAP_PROP_POS_MSEC, curr_time * 1000.0)
        success, frame = cap.read()

        if success:
            score = get_sharpness_score(frame)
            if score > best_score:
                best_score = score
                best_frame = frame
                best_ts = curr_time

        curr_time += step

    cap.release()

    # 3. Update State
    if best_frame is not None:
        return {
            "raw_frame": best_frame,
            "current_image": best_frame,
            "auto_save": best_frame,
            "timestamp": best_ts,
        }

    return {}
