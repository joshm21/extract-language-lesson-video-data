import argparse
from pathlib import Path
from functools import partial

import config
from core import timestamps
from core import pipeline


def get_video_ids(data_path):
    """Returns a list of folder names in the data directory."""
    return [d.name for d in Path(data_path).iterdir() if d.is_dir()]


def main():
    parser = argparse.ArgumentParser(
        description="Workshop Video Runner: Process single frames, full videos, or entire dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 1. Mode selection: Frame vs Video vs Directory
    parser.add_argument(
        "--mode",
        choices=["frame", "video", "all"],
        default="frame",
        help="'frame' = 1 frame from 1 video;, 'video' = n samples from one video, 'all' = process every folder in ./data"
    )

    # 2. Parameters
    parser.add_argument(
        "--id",
        type=str,
        default="14QbqkeiSDtU62syzgaOVXhXRzBJWhaNN",
        help="The video id to process (ignored if mode is 'all')"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of frames to extract per video (used in 'frame' and 'video' modes)"
    )

    args = parser.parse_args()

    # Determine which IDs to loop through
    video_ids = [args.id]
    if args.mode == "all":
        video_ids = get_video_ids("./data")
        print(
            f"--- Processing ALL videos in ./data ({len(video_ids)} found) ---")

    # Override config based on mode
    if args.mode == "frame":
        config.CONFIG["timer"] = partial(
            timestamps.uniform, count=1)
    elif args.mode == "video":
        config.CONFIG["timer"] = partial(
            timestamps.uniform, count=args.count)
    # For 'all', we keep the count set in config.py unless you want to override it here

    # 3. Main Loop
    for vid in video_ids:
        pipeline.process_video(vid, config.CONFIG)


if __name__ == "__main__":
    main()
