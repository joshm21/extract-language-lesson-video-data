import shutil
import numpy as np
import cv2
import csv
from pathlib import Path
from datetime import datetime
import logging
import config


def get_step_name(step):
    """Extracts 'module.function' name from a function or partial."""
    base_func = step.func if hasattr(step, 'func') else step
    module_short = base_func.__module__.split('.')[-1]
    return f"{module_short}.{base_func.__name__}"


def get_ts_str(ts):
    return f"{ts:.1f}"


class Runner:
    def __init__(self):
        # We pull these top-level definitions directly from config.py
        self.video_selector = config.VIDEOS
        self.timer_strategy = config.TIMESTAMPS
        self.frame_steps = config.FRAME_PIPELINE
        self.post_process = config.VIDEO_POST_PROCESS
        # 1. Setup Logging based on config
        log_level = getattr(config, "LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

        self.base_data_dir = Path("./data")
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    def run(self):
        self.logger.info(f"\n🚀 Starting Run {self.run_id}")

        # 1. SCOPE: Identify which videos to process
        discovery_state = {"data_dir": self.base_data_dir}
        video_ids = self.video_selector(discovery_state)["video_ids"]

        for v_id in video_ids:
            self._process_video(v_id)

    def _process_video(self, video_id):
        self.logger.info("-" * 40)
        self.logger.info(f"🎬 Processing Video: {video_id}")

        # Setup Video-specific Directories
        video_dir = self.base_data_dir / video_id
        artifacts_dir = video_dir / "artifacts" / f"run_{self.run_id}"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Archive config for reproducibility
        shutil.copy("config.py", artifacts_dir / "archived_config.py")

        # 2. STRATEGY: Get timestamps for this video
        timing_state = {"data_dir": self.base_data_dir, "video_id": video_id}
        timestamps = self.timer_strategy(timing_state)["timestamps"]

        all_video_crops = []  # Accumulator for deduplication

        # 3. FRAME LOOP
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

        # 4. POST-FRAME: Deduplicate all crops found in this video
        if all_video_crops and self.post_process:
            self.logger.debug(
                f"  ✨ Deduplicating {len(all_video_crops)} crops found across all frames.")
            unique_crops = self.post_process(all_video_crops)
            self._save_final_results(artifacts_dir, unique_crops)

    def _handle_auto_save(self, state, idx, step_name):
        """
        Saves images or 2D lists 'auto_save' key, then clears the key.
        Only saves if CONFIG.SAVE_ARTIFACTS != True
        """
        if not getattr(config, "SAVE_ARTIFACTS", True):
            return

        data = state.get("auto_save")
        if data is None:
            return

        base_name = f"{get_ts_str(state['timestamp'])}_{idx:02d}_{step_name}"
        save_path = state["artifacts_dir"] / base_name

        # 1. Handle Images
        # OpenCV images in Python are numpy arrays
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
        else:
            pass

        # Clear the field so it doesn't save twice
        state["auto_save"] = None

    def _save_final_results(self, artifacts_dir: Path, unique_crops: list):
        """
        Saves the de-duplicated crops with incrementing indices.
        """
        if not unique_crops:
            self.logger.warn("  ⚠️  No unique crops found to save.")
            return

        self.logger.info(
            f"  🏁 Finished: {len(unique_crops)} unique cards found.")
        self.logger.info(f"  📂 Folder: {str(artifacts_dir)}")
        for i, crop_img in enumerate(unique_crops):
            # Format: unique_00.jpg, unique_01.jpg, etc.
            filename = f"unique_{i:02d}.jpg"
            save_path = artifacts_dir / filename
            cv2.imwrite(str(save_path), crop_img)


if __name__ == "__main__":
    runner = Runner()
    runner.run()
    runner.run()
    runner.run()
    runner.run()
