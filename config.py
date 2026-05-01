from functools import partial

from core import (load, timestamps, extract, prepare,
                  detect, score, filter as filt, cluster, crop, dedupe)

# --- 1. DATA SCOPE ---
# Which videos are we running?
VIDEOS = load.test_video

# --- 2. SAMPLING STRATEGY ---
# How do we pick timestamps for each video?
TIMESTAMPS = partial(timestamps.every_n_seconds, n=20)

# --- 3. THE FRAME PIPELINE ---
# What happens to every single extracted frame?
FRAME_PIPELINE = [
    extract.at_current_timestamp,

    #    prepare.to_grayscale,
    #    partial(prepare.to_blurred, ksize=5),
    #    partial(prepare.at_adaptive_threshold,
    #            block_size=15, c_val=5),
    #    partial(prepare.do_closing, kernel_size=3),

    prepare.to_grayscale,
    partial(prepare.to_blurred, ksize=3),
    partial(prepare.at_canny_edges, low=50, high=200),
    partial(prepare.do_dilation, kernel_size=3, iterations=1),
    partial(prepare.do_closing, kernel_size=10),

    partial(detect.find_quads, min_area=50, epsilon=0.03),
    score.all_quads,

    # partial(filt.apply, prop=score.props.area, min=4000),
    partial(cluster.k_means, k=2, features=[
            score.props.area], select_max=score.props.area),

    crop.passed_quads
]

# --- 4. VIDEO POST-PROCESSING ---
# What happens once after all frames in a video are processed?
VIDEO_POST_PROCESS = partial(dedupe.get_unique, threshold=20)


# --- 5. EXECUTION SETTINGS ---
# Options from most to least verbose: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_LEVEL = "DEBUG"
# Set to False if you only want the final results, not intermediate steps
SAVE_ARTIFACTS = True
