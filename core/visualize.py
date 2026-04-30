import cv2
from pathlib import Path
import numpy as np
from typing import List


class PipelineVisualizer:
    def __init__(self, output_dir: Path, timestamp: str):
        self.output_dir = output_dir
        self.ts_str = timestamp
        self.step_index = 1

    def save(self, image, label: str):
        """
        Saves the image with an auto-incrementing prefix.
        Example: 001.2_01_original.jpg, 001.2_02_gray.jpg
        """
        filename = f"{self.ts_str}_{self.step_index:02d}_{label}.jpg"
        filepath = self.output_dir / filename
        cv2.imwrite(str(filepath), image)
        self.step_index += 1

    def save_list(self, images: List[np.ndarray], base_label: str):
        """Saves a series of images (like a waterfall) with sub-indices."""
        for i, img in enumerate(images):
            # Example: 001.2_05_waterfall_01.jpg
            filename = f"{self.ts_str}_{self.step_index:02d}_{base_label}_{i+1:02d}.jpg"
            cv2.imwrite(str(self.output_dir / filename), img)
        self.step_index += 1
