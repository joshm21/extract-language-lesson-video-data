import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class FilterStep:
    prop: str
    min: float = -float('inf')
    max: float = float('inf')


class CardCandidates:
    def __init__(self, quads: List[np.ndarray], scores: List[Dict[str, Any]], history: Optional[List] = None):
        """
        Initializes with quads and their pre-calculated scores.
        """
        # Ensure data integrity: every quad must have a corresponding score dictionary
        assert len(quads) == len(
            scores), f"Mismatch: {len(quads)} quads vs {len(scores)} scores"

        self.quads = quads
        self.scores = scores
        self.history = history if history is not None else []

    def apply_filters(self, image: np.ndarray, config: List[FilterStep]):
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
        print(f"Filter {step.prop}: {len(self.quads)} -> {len(passed_quads)}")

        return CardCandidates(passed_quads, passed_scores, new_history)


# --- VISUALIZATION ---
def visualize_waterfall(image: np.ndarray, candidates: CardCandidates) -> List[np.ndarray]:
    """Generates stages showing green (passed) and red (rejected) quads."""
    stages = []

    for i, (prop_name, passed, rejected) in enumerate(candidates.history):
        canvas = image.copy()
        overlay = canvas.copy()

        # Draw rejected quads in Red and passed in Green
        cv2.drawContours(overlay, rejected, -1, (0, 0, 255), 2)
        cv2.drawContours(canvas, passed, -1, (0, 255, 0), 3)

        cv2.addWeighted(overlay, 0.5, canvas, 0.7, 0, canvas)

        # Labels for the step and current counts
        cv2.putText(canvas, f"Step {i+1}: {prop_name}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, f"P: {len(passed)} | R: {len(rejected)}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        stages.append(canvas)

    return stages
