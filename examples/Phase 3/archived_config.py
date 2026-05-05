from functools import partial

from core import load
from core import timestamps
from core import extract
from core import prepare
from core import detect
from core import score
# from core import plot
# from core import filter as filt
from core import cluster
# from core import classify
from core import crop
from core import dedupe

# --- 1. DATA SCOPE ---
# Which videos are we running?
VIDEOS = load.test_video

# --- 2. SAMPLING STRATEGY ---
# How do we pick timestamps for each video?
TIMESTAMPS = partial(timestamps.uniform, count=1)

# --- 3. THE FRAME PIPELINE ---
# What happens to every single extracted frame?
FRAME_PIPELINE = [
    partial(extract.at_sharpest_in_window, window_seconds=0.5),

    prepare.to_grayscale,
    partial(prepare.to_blurred, ksize=3),
    partial(prepare.at_canny_edges, low=50, high=200),
    partial(prepare.do_dilation, kernel_size=3, iterations=1),

    partial(detect.find_quads, min_area=50, epsilon=0.03),
    score.all_quads,

    partial(cluster.k_means,
            k=2,
            features=[score.props.area,
                      score.props.aspect_ratio,
                      score.props.edge_density],
            select_max=score.props.area),

    crop.passed_quads
]

# --- 4. VIDEO POST-PROCESSING ---
# What happens once after all frames in a video are processed?
VIDEO_POST_PROCESS = partial(dedupe.get_unique, threshold=10)


# --- 5. EXECUTION SETTINGS ---
# Options from most to least verbose: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_LEVEL = "DEBUG"
# Set to False if you only want the final results, not intermediate steps
SAVE_ARTIFACTS = True
