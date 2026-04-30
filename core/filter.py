import cv2
import numpy as np
import csv
import dataclasses
from typing import List, Optional
from .score import ScoreCard


@dataclasses.dataclass
class QuadCandidate:
    """
    The single source of truth for a detected quad.
    Stores the geometry (points) and the metrics (score) in one object.
    """
    points: np.ndarray
    score: ScoreCard


class CardCandidates:
    def __init__(self, quads: List[QuadCandidate], history: Optional[List] = None, all_quads: Optional[List] = None):
        self.quads = quads
        self.history = history if history is not None else []
        # Store a reference to the initial list for final reporting
        self.all_quads = all_quads if all_quads is not None else quads

    def apply_filter_step(self, name: str, filter_func):
        """Runs a filter and returns a new container with the history updated."""
        passed, rejected = [], []

        for q in self.quads:
            if filter_func(q):  # IntelliSense works here!
                passed.append(q)
            else:
                rejected.append(q)

        new_history = self.history + [(name, passed, rejected)]
        return CardCandidates(passed, new_history, self.all_quads)

    def to_csv(self, filepath: str):
        """Exports all quads and their scores to a CSV file."""
        if not self.all_quads:
            return

        # 1. Dynamically build headers from coordinates and ScoreCard fields
        coord_headers = [f"{ax}{i}" for i in range(1, 5) for ax in ('x', 'y')]
        score_keys = [f.name for f in dataclasses.fields(ScoreCard)]
        history_headers = [f"step_{i+1}_{name}" for i,
                           (name, _, _) in enumerate(self.history)]

        fieldnames = coord_headers + score_keys + history_headers

        rows = []
        for q_obj in self.all_quads:
            # Add coordinates
            row = {coord_headers[i]: val for i,
                   val in enumerate(q_obj.points.reshape(-1))}

            # Add scores (no redundancy, directly from the dataclass)
            row.update(dataclasses.asdict(q_obj.score))

            # Add pass/fail status for every step in the history
            for i, (_, passed_list, _) in enumerate(self.history):
                # Using 'is' for identity check to be fast and precise
                is_pass = any(q_obj is p for p in passed_list)
                row[history_headers[i]] = "PASS" if is_pass else "FAIL"
            rows.append(row)

        with open(filepath, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def visualize_waterfall(image: np.ndarray, candidates: CardCandidates) -> List[np.ndarray]:
    """Generates the visual history of the filtering process."""
    stages = []
    COLOR_PASS, COLOR_REJECT = (0, 255, 0), (0, 0, 180)

    for i, (name, passed, rejected) in enumerate(candidates.history):
        canvas = image.copy()

        # Ensure points are int32 and have the shape (N, 1, 2) for OpenCV
        if passed:
            passed_contours = [np.int32(q.points).reshape(
                (-1, 1, 2)) for q in passed]
            cv2.drawContours(canvas, passed_contours, -
                             1, COLOR_PASS, 3)

        if rejected:
            overlay = canvas.copy()
            rejected_contours = [np.int32(q.points).reshape(
                (-1, 1, 2)) for q in rejected]
            cv2.drawContours(overlay, rejected_contours, -
                             1, COLOR_REJECT, 2)
            cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

        cv2.putText(canvas, f"Step {i+1}: {name}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        stages.append(canvas)
    return stages
