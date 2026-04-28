import cv2
import numpy as np
from typing import List, Callable, Dict, Any

# --- SECTION 1: THE ATOMIC FILTERS ---


def relative_area(image: np.ndarray, quad: np.ndarray, min_rel: float = 0.01, max_rel: float = 0.1) -> bool:
    """Checks if the card area is a reasonable percentage of the total frame."""
    frame_area = image.shape[0] * image.shape[1]
    quad_area = cv2.contourArea(quad)
    relative_area = quad_area / frame_area
    return min_rel <= relative_area <= max_rel


def convexity(image: np.ndarray, quad: np.ndarray) -> bool:
    """Cards are physical rectangles; their detected corners should form a convex shape."""
    return cv2.isContourConvex(quad)


def aspect_ratio(image: np.ndarray, quad: np.ndarray, target: float = 0.714, tolerance: float = 0.1) -> bool:
    """Checks if the width/height ratio matches a standard card (approx 0.714)."""
    x, y, w, h = cv2.boundingRect(quad)
    if h == 0:
        return False
    current = min(w, h) / max(w, h)
    return abs(target - current) <= tolerance


def extent(image: np.ndarray, quad: np.ndarray, min_extent: float = 0.7) -> bool:
    """Measures how much of the bounding box is filled by the shape (1.0 is a perfect rectangle)."""
    area = cv2.contourArea(quad)
    x, y, w, h = cv2.boundingRect(quad)
    extent = area / (w * h) if (w * h) > 0 else 0
    return extent > min_extent


def edge_density(image: np.ndarray, quad: np.ndarray, min_density: float = 0.05) -> bool:
    """High edge density usually indicates complex card art rather than a blank table."""
    # Note: In a workshop, students would use crop.py first or a simple ROI here
    x, y, w, h = cv2.boundingRect(quad)
    roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return False
    edges = cv2.Canny(roi, 50, 150)
    density = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1])
    return density > min_density


def color_variance(image: np.ndarray, quad: np.ndarray, min_variance: float = 10.0) -> bool:
    """Filters out monochrome objects (like white paper) by checking color variety."""
    x, y, w, h = cv2.boundingRect(quad)
    roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return False
    return float(np.mean(np.std(roi, axis=(0, 1)))) > min_variance


class CardCandidates:
    def __init__(self, image, quads):
        self.image = image
        self.quads = quads  # List of all detected quads

    def filter(self, func, *args, **kwargs):
        """Applies a filter function and returns a new CardCandidates object."""
        # This passes (image, quad, *args) to the function
        new_quads = [q for q in self.quads if func(
            self.image, q, *args, **kwargs)]
        print(f"Filter {func.__name__}: {len(self.quads)} -> {len(new_quads)}")
        return CardCandidates(self.image, new_quads)


def visualize_decisions(image: np.ndarray, all_quads: List[np.ndarray], filtered_quads: List[np.ndarray]) -> np.ndarray:
    """Draws ALL in Red, then overlaps SELECTED in Green."""
    output = image.copy()
    cv2.drawContours(output, all_quads, -1, (0, 0, 255), 2)
    cv2.drawContours(output, filtered_quads, -1, (0, 255, 0), 3)
    return output
