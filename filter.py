import cv2
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import csv


@dataclass
class FilterStep:
    prop: str
    min: float = -float('inf')
    max: float = float('inf')


class CardCandidates:
    def __init__(self, quads: List[np.ndarray], scores: List[Dict[str, float]],
                 history: Optional[List] = None,
                 all_quads: Optional[List] = None,
                 all_scores: Optional[List] = None):
        """
        Initializes with quads and their pre-calculated scores.
        """
        # Ensure data integrity: every quad must have a corresponding score dictionary
        assert len(quads) == len(
            scores), f"Mismatch: {len(quads)} quads vs {len(scores)} scores"

        self.quads = quads
        self.scores = scores
        self.history = history if history is not None else []

        # Keep track of the original population for the CSV
        self.all_quads = all_quads if all_quads is not None else quads
        self.all_scores = all_scores if all_scores is not None else scores

    def apply_filters(self, config: List[FilterStep]):
        """Executes the pipeline of FilterStep objects."""
        current = self
        for step in config:
            current = current.filter_by_score(step)
        return current

    def filter_by_score(self, step: FilterStep):
        """Filters the current candidates based on the provided FilterStep criteria."""
        passed_quads, passed_scores = [], []
        rejected_quads = []

        for q, stats in zip(self.quads, self.scores):

            val = stats.get(step.prop)

            if val is not None and step.min <= val <= step.max:

                passed_quads.append(q)
                passed_scores.append(stats)
            else:
                rejected_quads.append(q)

        # History only needs to store the quad references for pass/fail tracking
        new_history = self.history + \
            [(step.prop, passed_quads, rejected_quads)]

        # Pass the master lists (all_quads/all_scores) to the next object
        return CardCandidates(
            passed_quads,
            passed_scores,
            new_history,
            all_quads=self.all_quads,
            all_scores=self.all_scores
        )

    def to_csv(self, filepath: str):
        """Exports all original quads, their scores, and pass/fail history."""
        if not self.all_quads:
            return

        # 1. Headers
        coord_headers = [f"{axis}{i}" for i in range(
            1, 5) for axis in ('x', 'y')]
        score_headers = sorted(list(self.all_scores[0].keys()))
        history_headers = [f"step_{i+1}_{name}" for i,
                           (name, _, _) in enumerate(self.history)]

        fieldnames = coord_headers + score_headers + history_headers

        # 2. Build rows using the master lists
        rows = []
        for quad, stats in zip(self.all_quads, self.all_scores):
            # Flatten coords
            coords = quad.reshape(-1)
            row = {coord_headers[i]: val for i, val in enumerate(
                coords) if i < len(coord_headers)}

            # Add original scores
            row.update(stats)

            # 3. Check pass/fail status against each history snapshot
            for i, (_, passed_list, _) in enumerate(self.history):
                is_pass = any(np.array_equal(quad, p) for p in passed_list)
                row[history_headers[i]] = "PASS" if is_pass else "FAIL"

            rows.append(row)

        with open(filepath, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def visualize_waterfall(image: np.ndarray, candidates: CardCandidates) -> List[np.ndarray]:
    """Generates stages showing green (passed) and red (rejected) quads."""
    stages = []

    # Define cohesive styles
    COLOR_PASS = (0, 255, 0)      # Bright Green
    COLOR_REJECT = (0, 0, 180)    # Red
    COLOR_TITLE = (255, 255, 255)  # Pure White

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    TEXT_SCALE = 0.7
    TEXT_THICKNESS = 2

    for i, (prop_name, passed, rejected) in enumerate(candidates.history):
        canvas = image.copy()
        overlay = canvas.copy()

        # 1. Draw Contours using the unified color palette
        if passed:
            # Convert quads to int32 and ensure they are shaped correctly for OpenCV
            passed_to_draw = [np.int32(q).reshape((-1, 1, 2)) for q in passed]
            cv2.drawContours(canvas, passed_to_draw, -1, COLOR_PASS, 3)

        if rejected:
            # Convert rejected quads similarly
            rejected_to_draw = [np.int32(q).reshape(
                (-1, 1, 2)) for q in rejected]
            cv2.drawContours(overlay, rejected_to_draw, -1, COLOR_REJECT, 2)

        # Blend rejected overlay for a "ghosted" look
        cv2.addWeighted(overlay, 0.5, canvas, 0.7, 0, canvas)

        # 2. Draw Title (Step Name)
        cv2.putText(canvas, f"Step {i+1}: {prop_name}", (20, 40),
                    FONT, 1.0, COLOR_TITLE, TEXT_THICKNESS)

        # 3. Draw Pass/Reject Status Line
        pass_text = f"Pass: {len(passed)}"
        reject_text = f" | Reject: {len(rejected)}"

        # Position at y=110 to keep it clear of the top edge of the frame
        curr_y = 110

        # Draw "Pass" in Green
        cv2.putText(canvas, pass_text, (20, curr_y),
                    FONT, TEXT_SCALE, COLOR_PASS, TEXT_THICKNESS)

        # Calculate offset to place "Reject" immediately after "Pass"
        (text_width, _), _ = cv2.getTextSize(
            pass_text, FONT, TEXT_SCALE, TEXT_THICKNESS)

        # Draw "Reject" in Faded Red
        cv2.putText(canvas, reject_text, (20 + text_width, curr_y),
                    FONT, TEXT_SCALE, COLOR_REJECT, TEXT_THICKNESS)

        stages.append(canvas)

    return stages
