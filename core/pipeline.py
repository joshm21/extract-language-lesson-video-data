import csv
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import cv2
import numpy as np

import config


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def get_step_name(step: Callable) -> str:
    """Extracts 'module.function' name and appends partial arguments if they exist."""
    # 1. Identify the base function
    base_func = step.func if hasattr(step, 'func') else step
    module_short = base_func.__module__.split('.')[-1]
    name = f"{module_short}.{base_func.__name__}"

    # 2. If it's a partial, append the arguments to the name
    if hasattr(step, 'keywords') and step.keywords:
        # Create a string of key=val, filtered for simple types
        args_list = []
        for k, v in step.keywords.items():
            if isinstance(v, (str, int, float, bool)):
                # Sanitize value: replace slashes and backslashes with underscores
                safe_v = re.sub(r'[\\/]', '_', str(v))
                args_list.append(f"{k}={safe_v}")

        args_str = "~".join(args_list)
        if args_str:
            name = f"{name}({args_str})"

    return name


def get_ts_str(ts: float) -> str:
    """Formats a timestamp as a string to 1 decimal place."""
    return f"{ts:.1f}"


# ---------------------------------------------------------------------------
# Core Runner
# ---------------------------------------------------------------------------

class Runner:
    def __init__(self):
        # Top-level definitions pulled directly from config.py
        self.video_selector = config.VIDEOS
        self.timer_strategy = config.TIMESTAMPS
        self.frame_steps = config.FRAME_PIPELINE
        self.post_process = config.VIDEO_POST_PROCESS

        # Setup Logging based on config
        log_level = getattr(config, "LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(message)s',
            handlers=[logging.StreamHandler()],
            force=True
        )
        self.logger = logging.getLogger(__name__)

        self.base_data_dir = Path("./data")
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    def run(self) -> None:
        """Main execution entrypoint for the runner."""
        self.logger.info(f"\n🚀 Starting Run {self.run_id}")

        video_ids = self._discover_videos()
        all_results = []

        for v_id in video_ids:
            metrics = self._process_video(v_id)
            all_results.append((v_id, *metrics))

        self._print_final_report(all_results)

    # -----------------------------------------------------------------------
    # High-Level Pipeline Steps
    # -----------------------------------------------------------------------

    def _discover_videos(self) -> List[str]:
        """Identifies which videos to process based on config."""
        discovery_state = {"data_dir": self.base_data_dir}
        return self.video_selector(discovery_state)["video_ids"]

    def _process_video(self, video_id: str) -> Tuple[float, int, int, float, float]:
        """Orchestrates the directory setup, frame processing, and post-processing for a single video."""
        self.logger.info("-" * 40)
        self.logger.info(f"🎬 Processing Video: {video_id}")

        artifacts_dir = self._setup_video_directories(video_id)
        timestamps = self._get_timestamps(video_id)
        all_video_crops = self._process_frames(
            video_id, timestamps, artifacts_dir)

        return self._post_process_video(video_id, all_video_crops, artifacts_dir)

    def _setup_video_directories(self, video_id: str) -> Path:
        """Sets up video-specific directories and archives the configuration."""
        video_dir = self.base_data_dir / video_id
        artifacts_dir = video_dir / "artifacts" / f"run_{self.run_id}"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Archive config for reproducibility
        shutil.copy("config.py", artifacts_dir / "archived_config.py")
        return artifacts_dir

    def _get_timestamps(self, video_id: str) -> List[float]:
        """Retrieves timestamps for the given video using the timing strategy."""
        timing_state = {"data_dir": self.base_data_dir, "video_id": video_id}
        return self.timer_strategy(timing_state)["timestamps"]

    def _process_frames(self, video_id: str, timestamps: List[float], artifacts_dir: Path) -> List[Any]:
        """Iterates over timestamps and executes pipeline steps per frame."""
        all_video_crops = []

        for ts in timestamps:
            self.logger.debug(f"  🕒 @ {get_ts_str(ts)}s")

            # Initial state for this frame
            state = {
                "data_dir": self.base_data_dir,
                "video_id": video_id,
                "timestamp": ts,
                "artifacts_dir": artifacts_dir
            }

            for idx, step in enumerate(self.frame_steps, start=1):
                name = get_step_name(step)
                self.logger.debug(f"    [Step {idx:02d}] {name}")

                updates = step(state)
                if updates:
                    state.update(updates)

                self._handle_auto_save(state, idx, name)

            # Collect results for the post-frame deduplication
            if "crops" in state:
                all_video_crops.extend(state["crops"])

        return all_video_crops

    def _post_process_video(self, video_id: str, all_video_crops: List[Any], artifacts_dir: Path) -> Tuple[float, int, int, float, float]:
        """Deduplicates crops, saves results, and generates local accuracy reports."""
        metrics = (0.0, 0, 0, 0.0, 0.0)  # Default empty metrics

        if all_video_crops and self.post_process:
            self.logger.debug(
                f"  ✨ Deduplicating {len(all_video_crops)} crops found across all frames.")
            unique_crops = self.post_process(all_video_crops)

            self._save_final_results(artifacts_dir, unique_crops)
            metrics = self._calculate_accuracy(video_id, unique_crops)

            report_path = artifacts_dir / "results.txt"
            report_content = self._format_single_report(video_id, metrics)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)

            self.logger.info(f"  📝 Local report saved: {report_path}")

        return metrics

    # -----------------------------------------------------------------------
    # Artifact Management & Saving
    # -----------------------------------------------------------------------

    def _handle_auto_save(self, state: Dict[str, Any], idx: int, step_name: str) -> None:
        """
        Saves images or 2D lists mapped to the 'auto_save' key, then clears the key.
        Only saves if CONFIG.SAVE_ARTIFACTS != False
        """
        if not getattr(config, "SAVE_ARTIFACTS", True):
            return

        data = state.get("auto_save")
        if data is None:
            return

        base_name = f"{get_ts_str(state['timestamp'])}_{idx:02d}_{step_name}"
        save_path = state["artifacts_dir"] / base_name

        # 1. Handle Images (OpenCV images in Python are numpy arrays)
        if isinstance(data, np.ndarray):
            cv2.imwrite(str(f'{save_path}.jpg'), data)

        # 2. Handle 2D Lists (Standard Lib CSV)
        elif isinstance(data, list) and len(data) > 0:
            with open(f'{save_path}.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                # If the first element is a list, write all rows (2D list)
                if isinstance(data[0], list):
                    writer.writerows(data)
                # Otherwise, treat as a single row (1D list)
                else:
                    writer.writerow(data)

        # Clear the field so it doesn't save twice
        state["auto_save"] = None

    def _save_final_results(self, artifacts_dir: Path, unique_crops: list) -> None:
        """Saves the de-duplicated crops with incrementing indices."""
        if not unique_crops:
            self.logger.warn("  ⚠️  No unique crops found to save.")
            return

        self.logger.info(
            f"  🏁 Finished: {len(unique_crops)} unique cards found.")
        self.logger.info(f"  📂 Folder: {str(artifacts_dir)}")

        for i, crop_img in enumerate(unique_crops):
            filename = f"unique_{i:02d}.jpg"
            save_path = artifacts_dir / filename
            cv2.imwrite(str(save_path), crop_img)

    # -----------------------------------------------------------------------
    # Reporting & Metrics
    # -----------------------------------------------------------------------

    def _calculate_accuracy(self, video_id: str, found_crops: list) -> Tuple[float, int, int, float, float]:
        """
        Compares found crops against images in data/{video-id}/goal.
        Penalizes extra 'junk' images using Precision logic.
        """
        from core.dedupe import compute_phash, is_duplicate

        goal_dir = self.base_data_dir / video_id / "goal"
        if not goal_dir.exists():
            return 0.0, 0, len(found_crops), 0.0, 0.0

        goal_files = list(goal_dir.glob("*.jpg"))
        if not goal_files:
            return 0.0, 0, len(found_crops), 0.0, 0.0

        # 1. Generate hashes for comparison
        goal_hashes = [compute_phash(cv2.imread(str(p))) for p in goal_files]
        found_hashes = [compute_phash(img) for img in found_crops]

        # 2. Count Matches (True Positives)
        matches = 0
        for g_h in goal_hashes:
            if is_duplicate(g_h, found_hashes, threshold=15):
                matches += 1

        # 3. Calculate Metrics
        recall = (matches / len(goal_hashes)) * 100
        precision = (matches / len(found_hashes)) * 100 if found_hashes else 0
        f1 = 2 * (precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0

        return f1, len(goal_hashes), len(found_hashes), precision, recall

    def _format_single_report(self, video_id: str, metrics: Tuple) -> str:
        """Standardizes the string format for one video's performance."""
        f1, g_c, f_c, prec, rec = metrics
        width = 40
        header = f"{'Video ID':<{width}} | {'Goal':^6} | {'Found':^6} | {'Prec':^6} | {'Rec':^6} | {'F1 Score'}"
        row = f"{video_id:<{width}} | {g_c:^6} | {f_c:^6} | {prec:>5.1f}% | {rec:>5.1f}% | {f1:>8.2f}%"
        return f"{header}\n{'-' * len(header)}\n{row}\n"

    def _print_final_report(self, all_results: List[Tuple]) -> None:
        """Prints the final formatted aggregated report to the console."""
        width = 40
        header = f"{'Video ID':<{width}} | {'Goal':^6} | {'Found':^6} | {'Prec':^6} | {'Rec':^6} | {'F1 Score'}"

        print("\n" + "=" * len(header))
        print(f"📊 PIPELINE PERFORMANCE REPORT ({self.run_id})")
        print("-" * len(header))
        print(header)
        print("-" * len(header))

        total_f1 = 0
        for v_id, f1, g_c, f_c, prec, rec in all_results:
            print(
                f"{v_id:<{width}} | {g_c:^6} | {f_c:^6} | {prec:>5.1f}% | {rec:>5.1f}% | {f1:>8.2f}%")
            total_f1 += f1

        avg_accuracy = total_f1 / len(all_results) if all_results else 0
        print("-" * len(header))
        print(f"{'OVERALL PIPELINE F1 SCORE:':<{width + 30}} {avg_accuracy:>8.2f}%")
        print("=" * len(header))
