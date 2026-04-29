import cv2
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class FilterStep:
    prop: str
    min: float = -float('inf')
    max: float = float('inf')


class CardCandidates:
    def __init__(self, quads: List[np.ndarray], scores: List[Dict[str, float]], history: Optional[List] = None):
        """
        Initializes with quads and their pre-calculated scores.
        """
        # Ensure data integrity: every quad must have a corresponding score dictionary
        assert len(quads) == len(
            scores), f"Mismatch: {len(quads)} quads vs {len(scores)} scores"

        self.quads = quads
        self.scores = scores
        self.history = history if history is not None else []

    def apply_filters(self, config: List[FilterStep]):
        """Executes the pipeline of FilterStep objects."""
        current = self
        for step in config:
            current = current.filter_by_score(step)
        return current

    def filter_by_score(self, step: FilterStep):
        """Filters the current candidates based on the provided FilterStep criteria."""
        passed_quads = []
        passed_scores = []
        rejected_quads = []

        for q, stats in zip(self.quads, self.scores):
            val = stats.get(step.prop)

            # Check if value exists and falls within the min/max range
            if val is not None and step.min <= val <= step.max:
                passed_quads.append(q)
                passed_scores.append(stats)
            else:
                rejected_quads.append(q)

        # Log history snapshots for visualization
        new_history = self.history + \
            [(step.prop, passed_quads, rejected_quads)]

        return CardCandidates(passed_quads, passed_scores, new_history)


# --- VISUALIZATION ---
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
        # Passed quads get the bright green
        cv2.drawContours(canvas, passed, -1, COLOR_PASS, 3)
        # Rejected quads get the faded red
        cv2.drawContours(overlay, rejected, -1, COLOR_REJECT, 2)

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
