import cv2
from typing import List


def uniform(cap: cv2.VideoCapture, count: int = 5) -> List[float]:
    """
    Generates a list of timestamps evenly spaced across the entire video.

    Args:
        cap: The open cv2.VideoCapture object.
        count: The total number of timestamps to generate. 
               e.g., 3 will return [start, middle, end].

    Returns:
        A list of floats representing seconds.
    """
    if count <= 1:
        return [0.0]

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 1:
        return [0.0]

    # Subtract 1 frame to be safe
    duration = (total_frames - 1) / fps if fps > 0 else 0

    return [i * (duration / (count - 1)) for i in range(count)]


def every_n_seconds(cap: cv2.VideoCapture, n: float = 1.0) -> List[float]:
    """
    Generates a list of timestamps at a fixed interval until the end of the video.

    Args:
        cap: The open cv2.VideoCapture object.
        n: The interval in seconds between each timestamp.

    Returns:
        A list of floats representing seconds.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    timestamps = []
    current_time = 0.0

    while current_time <= duration:
        timestamps.append(round(current_time, 2))
        current_time += n

    return timestamps
