from pathlib import Path
from datetime import datetime
import shutil
import cv2

from . import visualize
from . import filter as filt


def process_video(video_id, config):
    print(f"processing video {video_id}")
    base_dir = Path(f"./data/{video_id}")

    # Format: YYYYMMDD-HHMMSS (e.g., 20260430-140934)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    artifacts_dir = base_dir / "artifacts" / f"run_{run_id}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy("config.py", artifacts_dir / "archived_config.py")

    cap = cv2.VideoCapture(str(base_dir / "video.mp4"))
    ts_list = config["timer"](cap)

    all_crops = []
    for ts in ts_list:
        new_crops = process_frame(artifacts_dir, cap, ts, config)
        all_crops.extend(new_crops)
    cap.release()  # done extracting frames; release video

    print(f'- deduplicating')
    deduped = config["deduper"](all_crops)
    for i, d in enumerate(deduped):
        cv2.imwrite(str(artifacts_dir / f'unique{i:02d}.jpg'), d)
    print(f'- found {len(deduped)} unique cards in video')


def process_frame(artifacts_dir, cap, ts, config):
    ts_str = f"{ts:05.1f}"
    print(f"- frame at {ts_str}")
    viz = visualize.PipelineVisualizer(artifacts_dir, ts_str)

    print(f"  * extracting")
    raw_frame = config["extractor"](cap, ts)
    if raw_frame is None:
        return []
    viz.save(raw_frame, "raw")

    print(f"  * processing")
    current_img = raw_frame
    for step in config["processing"]:
        current_img = step["func"](current_img)
        viz.save(current_img, step["name"])

    print(f'  * detecting quads')
    raw_quads = config["detector"](current_img)
    print(f'    > {len(raw_quads)} quads detected')

    print(f'  * scoring quads')
    quad_candidates = []
    for q in raw_quads:
        score_card = config["scorer"](raw_frame, q)
        candidate = filt.QuadCandidate(points=q, score=score_card)
        quad_candidates.append(candidate)

    print(f'  * filtering quads')
    results = filt.CardCandidates(quad_candidates)
    for step in config["filters"]:
        pre_filter_count = len(results.quads)
        results = results.apply_filter_step(step["name"], step["func"])
        post_filter_count = len(results.quads)
        print(
            f'    > {step["name"]}: {pre_filter_count} -> {post_filter_count}')
    viz.save_list(filt.visualize_waterfall(raw_frame, results), "results")
    results.to_csv(str(artifacts_dir / f"{ts_str}-data.csv"))

    print(f'  * cropping cards')
    crops = config["cropper"](raw_frame, [q.points for q in results.quads])

    return crops
