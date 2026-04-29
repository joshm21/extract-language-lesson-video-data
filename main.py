from pathlib import Path
import cv2

import extract
import detect
import score
import filter as filt
import crop
import dedupe


def process_video(video_id, timestamps):
    print(f"Processing {video_id}")
    base_dir = Path(f"./data/{video_id}")
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # clean out artifacts
    for item in artifacts_dir.iterdir():
        if item.is_file():
            item.unlink()

    cap = cv2.VideoCapture(str(base_dir / "video.mp4"))

    all_crops = []
    for ts in timestamps:
        new_crops = process_frame(artifacts_dir, cap, ts)
        all_crops.extend(new_crops)

    # done extracting frames; release video
    cap.release()

    # dedupe crops from this video from any frame
    seen_hashes = []
    for img_crop in all_crops:
        hash = dedupe.compute_phash(img_crop)
        if dedupe.is_duplicate(hash, seen_hashes, threshold=20):
            continue
        seen_hashes.append(hash)
        filepath = str(artifacts_dir / f'unique{len(seen_hashes):03d}.jpg')
        cv2.imwrite(filepath, img_crop)  # save unique crop
    print(f'found {len(seen_hashes)} unique cards')


def process_frame(artifacts_dir, cap, ts):
    ts_str = f"{ts:05.1f}"  # eg 001.2 (seconds)
    print(f"...frame {ts_str}")
    frame_crops = []

    # extract
    frame = extract.extract_frame_at_time(cap, ts)
    if frame is None:
        return []
    # save frame
    cv2.imwrite(str(artifacts_dir / f'{ts_str}-frame.jpg'), frame)

    # detect
    # raw_quads = detect.detect_basic(frame, thresh_val=150)
    raw_quads = detect.sweep_detect_cards(frame)

    # score
    scores = [score.score_quad(frame, q) for q in raw_quads]

    # filter
    candidates = filt.CardCandidates(raw_quads, scores)
    filter_config = [
        filt.FilterStep(prop="relative_area", min=0.002, max=0.1),
        filt.FilterStep(prop="convexity", min=0.9)
    ]
    results = candidates.apply_filters(filter_config)

    # save results csv and visualization waterfall
    results.to_csv(str(artifacts_dir / f"{ts_str}-data.csv"))
    gallery = filt.visualize_waterfall(frame, results)
    for i, stage_img in enumerate(gallery):
        cv2.imwrite(
            str(artifacts_dir / f"{ts_str}-vis-{i+1:02d}.jpg"), stage_img)

    for quad in results.quads:
        card_crop = crop.get_perspective_transform(
            frame, detect.order_points(quad))
        frame_crops.append(card_crop)

    return frame_crops


if __name__ == "__main__":
    video_id = "14QbqkeiSDtU62syzgaOVXhXRzBJWhaNN"
    timestamps = list(range(0, 60, 10))  # [0, 2, 4, 6, 8]
    process_video(video_id, timestamps)
