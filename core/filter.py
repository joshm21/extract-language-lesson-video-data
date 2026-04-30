from functools import partial
from typing import Dict
from . import visualize as viz

# --- Generic Filter Engine ---


def _apply_threshold(
    state: Dict,
    metric_name: str,
    min: float = -float('inf'),
    max: float = float('inf')
) -> Dict:
    """
    Core logic: Filters quads based on the previous pass and generates visualization.
    """
    all_quads = state.get("quads", [])
    all_scores = state.get("scores", [])
    raw_image = state.get("raw_frame")

    # Get indices that survived previous filters, or start with all
    prev_indices = state.get("passed_indices", list(range(len(all_quads))))

    current_passed = []
    for idx in prev_indices:
        score_val = all_scores[idx].get(metric_name, 0)
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
        "auto_save": viz_image  # Runner handles the saving
    }

# --- Manually Defined Wrappers for Perfect Intellisense ---


def _make_filter(name):
    """Helper to create a partial and name it correctly for the runner."""
    p = partial(_apply_threshold, metric_name=name)
    p.__name__ = name  # This is the key for Intellisense and naming
    p.__module__ = __name__  # Force module to be filter
    return p


aspect_ratio = _make_filter("aspect_ratio")
extent = _make_filter("extent")
solidity = _make_filter("solidity")
convexity = _make_filter("convexity")
area = _make_filter("area")
relative_area = _make_filter("relative_area")
equivalent_diameter = _make_filter("equivalent_diameter")
orientation = _make_filter("orientation")
color_variance = _make_filter("color_variance")
saturation_average = _make_filter("saturation_average")
mean_intensity = _make_filter("mean_intensity")
edge_density = _make_filter("edge_density")
