from typing import Dict
from . import visualize as viz


def apply(
    state: Dict,
    prop: str,
    min: float = -float('inf'),
    max: float = float('inf')
) -> Dict:
    """
    Filters quads based on a specific metric and generates visualization.

    Args:
        state: Current frame state.
        prop: The prop to filter by (use score.props).
        min_val: Minimum threshold.
        max_val: Maximum threshold.
    """
    all_quads = state.get("quads", [])
    all_scores = state.get("scores", [])
    raw_image = state.get("raw_frame")

    # Get indices that survived previous filters, or start with all
    prev_indices = state.get("passed_indices", list(range(len(all_quads))))

    current_passed = []
    for idx in prev_indices:
        score_val = all_scores[idx].get(prop, 0)
        if min <= score_val <= max:
            current_passed.append(idx)

    # Calculate who was "killed" in this specific step for the red group
    rejected_this_step = [i for i in prev_indices if i not in current_passed]

    # --- Group for visualize.py ---
    # Passing 2 groups: (quads, labels, color)
    green_group = {
        'quads': [all_quads[i] for i in current_passed],
        'labels': [f"{i}" for i in current_passed],
        'color': (0, 255, 0)
    }

    red_group = {
        'quads': [all_quads[i] for i in rejected_this_step],
        'labels': [f"{i}" for i in rejected_this_step],
        'color': (0, 0, 255)
    }

    viz_image = viz.draw_multiple_quad_groups(
        raw_image, [green_group, red_group])

    return {
        "passed_indices": current_passed,
        "auto_save": viz_image
    }
