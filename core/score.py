import cv2
import numpy as np
from typing import Dict, Callable
from dataclasses import dataclass


# THE AUTOCOMPLETE CLASS (For config.py usage)

@dataclass(frozen=True)
class props:
    """Namespace for metric names to provide autocomplete and definitions."""

    aspect_ratio: str = "aspect_ratio"
    """Ratio of width to height of the bounding rect."""

    extent: str = "extent"
    """Ratio of contour area to bounding rectangle area."""

    solidity: str = "solidity"
    """Ratio of contour area to convex hull area."""

    convexity: str = "convexity"
    """Returns 1.0 if the shape is convex, 0.0 otherwise."""

    area: str = "area"
    """Returns the absolute area of the contour in pixels."""

    relative_area: str = "relative_area"
    """Returns the quad area as a percentage of the total frame area."""

    equivalent_diameter: str = "equivalent_diameter"
    """Diameter of the circle with the same area as the contour."""

    orientation: str = "orientation"
    """Returns the rotation angle using the Minimum Area Rectangle."""

    color_variance: str = "color_variance"
    """Mean of standard deviations across color channels."""

    saturation_average: str = "saturation_average"
    """Average saturation within the quad (detects color vs grayscale)."""

    mean_intensity: str = "mean_intensity"
    """Calculates average grayscale intensity within the contour mask."""

    edge_density: str = "edge_density"
    """Returns the density of Canny edges within the quad bounding box."""


# A registry to hold all functions that return a score
METRIC_REGISTRY: Dict[str, Callable] = {}


def metric(func):
    """Decorator to register a function as a score metric."""
    METRIC_REGISTRY[func.__name__.replace("get_", "")] = func
    return func


def all_quads(state: Dict) -> Dict:
    """Processes all registered scoring metrics for all quads."""
    image = state.get("raw_frame")
    quads = state.get("quads", [])

    if image is None or not quads:
        return {"scores": []}

    metric_names = sorted(METRIC_REGISTRY.keys())
    header = ["quad_index"] + metric_names

    scores_for_state = []
    rows_for_csv = [header,]
    for i, quad in enumerate(quads):
        # Dictionary comprehension: runs every @metric function automatically
        quad_scores = {name: fn(image, quad)
                       for name, fn in METRIC_REGISTRY.items()}
        scores_for_state.append(quad_scores)
        row = [i] + [quad_scores[name] for name in metric_names]
        rows_for_csv.append(row)

    return {"scores": scores_for_state, "auto_save": rows_for_csv}


@metric
def get_aspect_ratio(image: np.ndarray, quad: np.ndarray) -> float:
    """Returns the ratio of width to height of the bounding rect."""
    x, y, w, h = cv2.boundingRect(quad)
    if h == 0:
        return 0.0
    return float(w) / h


@metric
def get_extent(image: np.ndarray, quad: np.ndarray) -> float:
    """Returns the ratio of contour area to bounding rectangle area."""
    area = cv2.contourArea(quad)
    x, y, w, h = cv2.boundingRect(quad)
    rect_area = w * h
    if rect_area == 0:
        return 0.0
    return float(area) / rect_area


@metric
def get_solidity(image: np.ndarray, quad: np.ndarray) -> float:
    """Returns the ratio of contour area to convex hull area."""
    area = cv2.contourArea(quad)
    hull = cv2.convexHull(quad)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0.0
    return float(area) / hull_area


@metric
def get_convexity(image: np.ndarray, quad: np.ndarray) -> float:
    """Returns 1.0 if the shape is convex, 0.0 otherwise."""
    return 1.0 if cv2.isContourConvex(quad) else 0.0


@metric
def get_area(image: np.ndarray, quad: np.ndarray) -> float:
    """Returns the absolute area of the contour in pixels."""
    return float(cv2.contourArea(quad))


@metric
def get_relative_area(image: np.ndarray, quad: np.ndarray) -> float:
    """Returns the quad area as a percentage of the total frame area."""
    frame_area = image.shape[0] * image.shape[1]
    quad_area = cv2.contourArea(quad)
    if frame_area == 0:
        return 0.0
    return float(quad_area / frame_area)


@metric
def get_equivalent_diameter(image: np.ndarray, quad: np.ndarray) -> float:
    """Returns the diameter (in pixels) of the circle with the same area as the contour."""
    area = cv2.contourArea(quad)
    return float(np.sqrt(4 * area / np.pi))


@metric
def get_orientation(image: np.ndarray, quad: np.ndarray) -> float:
    """Returns the rotation angle using the Minimum Area Rectangle."""
    # rect is ((center_x, center_y), (width, height), angle)
    rect = cv2.minAreaRect(quad)
    return float(rect[2])


@metric
def get_color_variance(image: np.ndarray, quad: np.ndarray) -> float:
    """Measures color variety by checking the mean of standard deviations across color channels."""
    x, y, w, h = cv2.boundingRect(quad)
    roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.0
    return float(np.mean(np.std(roi, axis=(0, 1))))


@metric
def get_saturation_average(image: np.ndarray, quad: np.ndarray) -> float:
    """
    Calculates the average saturation within the quad.
    Grayscale images will have very low saturation.
    Colored images will have significantly higher saturation.
    """
    x, y, w, h = cv2.boundingRect(quad)
    roi = image[y:y+h, x:x+w]

    if roi.size == 0:
        return 0.0

    # Convert ROI to HSV color space
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Split channels (Hue, Saturation, Value)
    _, s_channel, _ = cv2.split(hsv_roi)

    # Return the average of the Saturation channel
    return float(np.mean(s_channel))


@metric
def get_mean_intensity(image: np.ndarray, quad: np.ndarray) -> float:
    """Calculates average grayscale intensity within the contour mask."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [quad], -1, 255, -1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = cv2.mean(gray, mask=mask)[0]
    return float(mean_intensity)


@metric
def get_edge_density(image: np.ndarray, quad: np.ndarray) -> float:
    """Returns the density of Canny edges within the quad bounding box."""
    x, y, w, h = cv2.boundingRect(quad)
    roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.0
    edges = cv2.Canny(roi, 50, 150)
    return float(np.sum(edges > 0) / (roi.shape[0] * roi.shape[1]))
