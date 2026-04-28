import cv2
import numpy as np
from typing import Tuple, Optional


def get_clear_frame(video_path: str, start_time: float, end_time: float, search_window: float = 1.0) -> Tuple[Optional[np.ndarray], float]:
    """
    Searches a specific video segment to find the frame with the highest Laplacian variance (sharpness).

    Args:
        video_path: Path to the source MP4 file.
        start_time: Start of the segment in seconds.
        end_time: End of the segment in seconds.
        search_window: Duration in seconds to search around the midpoint.

    Returns:
        A tuple of (Best Frame Image Array, Timestamp of that frame).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    midpoint = (start_time + end_time) / 2
    search_start = max(start_time, midpoint - (search_window / 2))
    search_end = min(end_time, midpoint + (search_window / 2))

    best_frame = None
    best_laplacian = -1.0
    best_timestamp = midpoint

    current_frame_idx = int(search_start * fps)
    end_frame_idx = int(search_end * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)

    while current_frame_idx <= end_frame_idx:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if laplacian_var > best_laplacian:
            best_laplacian = laplacian_var
            best_frame = frame
            best_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        current_frame_idx += 1

    cap.release()
    return best_frame, best_timestamp
