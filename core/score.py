import cv2
import numpy as np
import inspect
import sys
from typing import Dict
from dataclasses import dataclass


@dataclass
class ScoreCard:
    """
    Container for quad metrics to provide IntelliSense in config.py.
    """
    aspect_ratio: float
    extent: float
    solidity: float
    convexity: float
    area: float
    relative_area: float
    equivalent_diameter: float
    orientation: float
    color_variance: float
    saturation: float
    mean_intensity: float
    edge_density: float


def score_quad(image: np.ndarray, quad: np.ndarray) -> ScoreCard:
    """
    Explicitly calculates metrics and returns a structured ScoreCard.
    """
    return ScoreCard(
        aspect_ratio=get_aspect_ratio(image, quad),
        extent=get_extent(image, quad),
        solidity=get_solidity(image, quad),
        convexity=get_convexity(image, quad),
        area=get_area(image, quad),
        relative_area=get_relative_area(image, quad),
        equivalent_diameter=get_equivalent_diameter(image, quad),
        orientation=get_orientation(image, quad),
        color_variance=get_color_variance(image, quad),
        saturation=get_saturation_average(image, quad),
        mean_intensity=get_mean_intensity(image, quad),
        edge_density=get_edge_density(image, quad)
    )


def get_aspect_ratio(image: np.ndarray, quad: np.ndarray) -> float:
    """Returns the ratio of width to height of the bounding rect."""
    x, y, w, h = cv2.boundingRect(quad)
    if h == 0:
        return 0.0
    return float(w) / h


def get_extent(image: np.ndarray, quad: np.ndarray) -> float:
    """Returns the ratio of contour area to bounding rectangle area."""
    area = cv2.contourArea(quad)
    x, y, w, h = cv2.boundingRect(quad)
    rect_area = w * h
    if rect_area == 0:
        return 0.0
    return float(area) / rect_area


def get_solidity(image: np.ndarray, quad: np.ndarray) -> float:
    """Returns the ratio of contour area to convex hull area."""
    area = cv2.contourArea(quad)
    hull = cv2.convexHull(quad)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0.0
    return float(area) / hull_area


def get_convexity(image: np.ndarray, quad: np.ndarray) -> float:
    """Returns 1.0 if the shape is convex, 0.0 otherwise."""
    return 1.0 if cv2.isContourConvex(quad) else 0.0


def get_area(image: np.ndarray, quad: np.ndarray) -> float:
    """Returns the absolute area of the contour in pixels."""
    return float(cv2.contourArea(quad))


def get_relative_area(image: np.ndarray, quad: np.ndarray) -> float:
    """Returns the quad area as a percentage of the total frame area."""
    frame_area = image.shape[0] * image.shape[1]
    quad_area = cv2.contourArea(quad)
    if frame_area == 0:
        return 0.0
    return float(quad_area / frame_area)


def get_equivalent_diameter(image: np.ndarray, quad: np.ndarray) -> float:
    """Returns the diameter (in pixels) of the circle with the same area as the contour."""
    area = cv2.contourArea(quad)
    return float(np.sqrt(4 * area / np.pi))


def get_orientation(image: np.ndarray, quad: np.ndarray) -> float:
    """Returns the rotation angle using the Minimum Area Rectangle."""
    # rect is ((center_x, center_y), (width, height), angle)
    rect = cv2.minAreaRect(quad)
    return float(rect[2])


def get_color_variance(image: np.ndarray, quad: np.ndarray) -> float:
    """Measures color variety by checking the mean of standard deviations across color channels."""
    x, y, w, h = cv2.boundingRect(quad)
    roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.0
    return float(np.mean(np.std(roi, axis=(0, 1))))


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


def get_mean_intensity(image: np.ndarray, quad: np.ndarray) -> float:
    """Calculates average grayscale intensity within the contour mask."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [quad], -1, 255, -1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = cv2.mean(gray, mask=mask)[0]
    return float(mean_intensity)


def get_edge_density(image: np.ndarray, quad: np.ndarray) -> float:
    """Returns the density of Canny edges within the quad bounding box."""
    x, y, w, h = cv2.boundingRect(quad)
    roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.0
    edges = cv2.Canny(roi, 50, 150)
    return float(np.sum(edges > 0) / (roi.shape[0] * roi.shape[1]))
