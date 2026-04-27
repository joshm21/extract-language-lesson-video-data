import cv2
import json
import matplotlib.pyplot as plt
from pathlib import Path

import extract_segment_frames as extractor
import detect_cards_in_frame as detector
import filter_picture_cards as filterer
import crop_cards as cropper
import deduplicate_images as deduper


def save_debug_visual(image, all_cards, winners, output_path):
    """Saves a plot showing detection vs selection for debugging."""
    plt.figure(figsize=(12, 8))
    display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw all candidates (Yellow)
    for card in all_cards:
        cv2.polylines(display_img, [card["raw_quad"]], True, (255, 255, 0), 2)

    # Draw winners (Green)
    for card in winners:
        cv2.polylines(display_img, [card["raw_quad"]], True, (0, 255, 0), 5)

    plt.imshow(display_img)
    plt.title(
        f"Candidates: {len(all_cards)} (Yellow) | Filtered Winners: {len(winners)} (Green)")
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()


def run_pipeline(video_id: str, debug: bool = False):
    print(f"Starting pipeline for {video_id}")
    work_dir = Path(f"data/{video_id}")
    cache_dir = work_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    output_dir = work_dir / "unique_cards"
    output_dir.mkdir(exist_ok=True)
    debug_dir = work_dir / "debug"
    if debug:
        debug_dir.mkdir(exist_ok=True)

    cropped_images = []
    print('- loading segments')
    with open(work_dir / "transcript.json", 'r') as f:
        segments = json.load(f)['segments']

    for i, seg in enumerate(segments):
        print(f'- segment {i+1}')

        cache_pattern = f"seg_{seg['start']}_{seg['end']}_*.jpg"
        existing_cache = list(cache_dir.glob(cache_pattern))
        if existing_cache:
            print('  * loading frame from cache')
            cache_file = existing_cache[0]
            frame = cv2.imread(str(cache_file))
            # Extract timestamp: get the part between the last '_' and the '.jpg'
            ts = float(cache_file.stem.split('_')[-1])
        else:
            print('  * extracting clear frame (slow run)')
            frame, ts = extractor.get_clear_frame(
                str(work_dir / "video.mp4"), seg['start'], seg['end'])

            if frame is not None:
                # Save to cache with the found timestamp in the name
                cache_path = cache_dir / \
                    f"seg_{seg['start']}_{seg['end']}_{ts:.3f}.jpg"
                cv2.imwrite(str(cache_path), frame)

        if frame is None:
            continue

        all_candidates = detector.get_all_candidates(frame)
        print(f'  * detected {len(all_candidates)} cards in frame')
        if debug:
            # We strip the 'raw_quad' (numpy array) because it isn't JSON serializable
            serializable_data = []
            for c in all_candidates:
                item = {k: v for k, v in c.items() if k != 'raw_quad'}
                serializable_data.append(item)
            with open(debug_dir / f"{ts:.3f}_all_candidates.json", 'w') as f:
                json.dump(serializable_data, f, indent=4)

        winners = filterer.filter_picture_cards(all_candidates)
        print(f'  * filtered down to {len(winners)} picture cards')
        if debug:
            save_debug_visual(frame, all_candidates, winners,
                              debug_dir / f"{ts:.3f}_visualize.jpg")

        # Deskew and Crop
        print(f'  * deskewing and cropping')
        for i, card in enumerate(winners):
            cropped = cropper.deskew_and_crop(frame, card['coords'])
            cropped_images.append(cropped)

    print(f'- found {len(cropped_images)} card images across all frames')

    unique_images = deduper.deduplicate_images(cropped_images)
    print(f'  * {len(unique_images)} are unique')

    print(f'- emptying output directory and saving most recent unique cards')
    if output_dir.exists():
        for file in output_dir.iterdir():
            if file.is_file():
                file.unlink()
    for i, img in enumerate(unique_images):
        output_path = output_dir / f"card_{i+1:02d}.jpg"
        cv2.imwrite(str(output_path), img)


if __name__ == "__main__":
    run_pipeline("14QbqkeiSDtU62syzgaOVXhXRzBJWhaNN", debug=True)
