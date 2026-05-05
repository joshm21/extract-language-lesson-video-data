import cv2
import numpy as np
from typing import Dict


def get_sharpness_score(frame: np.ndarray) -> float:
    """
    Optimized sharpness score: downscales the image before
    calculating Laplacian variance to save CPU cycles.
    """
    if frame is None:
        return 0.0

    # Resize to a smaller width to speed up gray conversion and Laplacian
    # 320px or 640px is usually plenty for a sharpness delta check
    h, w = frame.shape[:2]
    scale = 320 / float(w)
    small_gray = cv2.cvtColor(
        cv2.resize(frame, (0, 0), fx=scale, fy=scale,
                   interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY
    )
    return cv2.Laplacian(small_gray, cv2.CV_64F).var()


def at_sharpest_in_window(state: Dict, window_seconds: float = 0.5) -> Dict:
    """
    High-performance sharpness window search.
    Uses frame grabbing to skip unnecessary decodes.
    """
    data_dir = state["data_dir"]
    video_id = state["video_id"]
    target_ts = state["timestamp"]

    video_path = data_dir / video_id / "video.mp4"
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Failed to open video at: {video_path}")  # Debug line
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    last_index = int(total_frames - 1)

    # Calculate frame-based bounds
    start_frame = int(max(0, target_ts - window_seconds) * fps)
    end_frame = int(min(last_index, (target_ts + window_seconds) * fps))

    # Sampling step in frames (e.g., every 3rd frame for ~30fps video is ~0.1s)
    frame_step = max(1, int(fps * 0.1))

    best_frame = None
    best_score = -1.0
    best_ts = target_ts

    # Seek ONCE to the beginning of the window
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_f = start_frame
    # 2. Ensure we don't exceed the last valid index[cite: 1]
    while current_f <= end_frame:
        # Use grab() to skip frames we don't need to score
        # This is much faster than cap.set() or cap.read()
        if current_f % frame_step == 0:
            success, frame = cap.read()
            if not success:
                # Exit if we hit the end of the stream prematurely[cite: 1]
                break

            score = get_sharpness_score(frame)
            if score > best_score:
                best_score = score
                best_frame = frame
                best_ts = current_f / fps
        else:
            # grab() also advances the cursor by 1 frame
            success = cap.grab()
            if not success:
                break

        current_f += 1

    cap.release()

    if best_frame is not None:
        return {
            "raw_frame": best_frame,
            "current_image": best_frame,
            "auto_save": best_frame,
            "timestamp": best_ts,
        }

    return {}
