from pathlib import Path
import cv2

import visualize
import extract
import prepare
import mask
import detect
import score
import filter as filt
import crop
import dedupe


TEST_VIDEO_ID = "14QbqkeiSDtU62syzgaOVXhXRzBJWhaNN"
DEBUG = True


def test_one_frame():
    process_video(TEST_VIDEO_ID, [1.0])


def test_one_video():
    timestamps = list(range(0, 60, 10))  # [0, 2, 4, 6, 8]
    process_video(TEST_VIDEO_ID, timestamps)


def process_all_videos():
    pass


def process_video(video_id, timestamps):
    print(f"processing video {video_id}")
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
    dedupe.save_unique(artifacts_dir, all_crops, threshold=5)


def process_frame(artifacts_dir, cap, ts):
    ts_str = f"{ts:05.1f}"
    viz = visualize.PipelineVisualizer(artifacts_dir, ts_str)
    print(f"processing frame {ts_str}")

    # extract
    frame = extract.extract_frame_at_time(cap, ts)
    if frame is None:
        return []
    viz.save(frame, "raw")

    # pre-process
    gray = prepare.get_grayscale(frame)
    viz.save(gray, "gray")
    blurred = prepare.apply_blur(gray, ksize=7)
    viz.save(blurred, "blurred")

    # mask
    binary_image = mask.canny(blurred, low=50, high=150)
    viz.save(binary_image, "canny-mask")

    # post-process
    # Use 'close' to bridge gaps in the card borders (important for #11)
    binary_image = prepare.apply_close(binary_image, kernel_size=3)
    # Use 'dilate' to ensure thin edges are thick enough to be detected (important for #4)
    binary_image = prepare.apply_dilate(
        binary_image, kernel_size=3, iterations=1)
    viz.save(binary_image, "post-processed-mask")

    # detect
    raw_quads = detect.get_quads(binary_image, min_area=300, epsilon=0.02)

    # score
    scores = [score.score_quad(frame, q) for q in raw_quads]

    # filter
    candidates = filt.CardCandidates(raw_quads, scores)
    filter_config = [
        filt.FilterStep(prop="relative_area", min=0.001, max=0.1),
        filt.FilterStep(prop="convexity", min=0.9),
        filt.FilterStep(prop="saturation_average", min=20),
    ]
    results = candidates.apply_filters(filter_config)

    # save results csv and visualization waterfall
    results.to_csv(str(artifacts_dir / f"{ts_str}-data.csv"))
    filter_waterfall = filt.visualize_waterfall(frame, results)
    viz.save_list(filter_waterfall, "filter")

    return crop.all(frame, results.quads)


if __name__ == "__main__":
    test_one_frame()
