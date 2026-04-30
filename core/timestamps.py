import cv2
from typing import List, Dict


def _get_video_duration(data_dir, video_id):
    """Internal helper to get duration without leaving handles open."""
    video_path = data_dir / video_id / "video.mp4"
    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = (total_frames - 1) / fps if fps > 0 else 0
    cap.release()
    return duration


def uniform(state: Dict, count: int = 5) -> Dict:
    """Generates evenly spaced timestamps across the video."""
    duration = _get_video_duration(state["data_dir"], state["video_id"])

    if count <= 1:
        ts_list = [0.0]
    else:
        ts_list = [i * (duration / (count - 1)) for i in range(count)]

    return {"timestamps": ts_list}


def every_n_seconds(state: Dict, n: float = 1.0) -> Dict:
    """Generates timestamps at fixed intervals until the video ends."""
    duration = _get_video_duration(state["data_dir"], state["video_id"])

    ts_list = []
    current_time = 0.0
    while current_time <= duration:
        ts_list.append(round(current_time, 2))
        current_time += n

    return {"timestamps": ts_list}


def at(state: Dict, seconds: float = 0.0) -> Dict:
    """Focuses the pipeline on a single specific timestamp."""
    return {"timestamps": [float(seconds)]}
