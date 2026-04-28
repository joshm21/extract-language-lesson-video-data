import cv2
import numpy as np
from typing import List, Dict, Any


def score_edge_density(gray_crop: np.ndarray) -> float:
    """Calculates density of edges; higher usually means complex art."""
    edges = cv2.Canny(gray_crop, 50, 150)
    return float(np.sum(edges > 0) / (gray_crop.shape[0] * gray_crop.shape[1]))


def score_color_variance(rgb_crop: np.ndarray) -> float:
    """Calculates variety of colors in the crop."""
    return float(np.mean(np.std(rgb_crop, axis=(0, 1))))


def score_aspect_ratio_fit(w: int, h: int) -> float:
    """Measures closeness to a $2.5 \times 3.5$ card ratio (approx 0.714)."""
    target = 0.714
    current = float(w) / h if h != 0 else 0
    if current > 1:
        current = 1 / current
    return float(1.0 - abs(target - current))


def _is_sane_quad(approx: np.ndarray, total_area: int) -> bool:
    """Checks if a contour is convex, fits area bounds, has a reasonable aspect ratio, and extents."""
    if not cv2.isContourConvex(approx):
        return False

    # Area filter: Discard anything too small or too large
    area = cv2.contourArea(approx)
    relative_area = area / total_area
    if not (0.001 < relative_area < 0.07):
        return False

    x, y, w, h = cv2.boundingRect(approx)

    short_side = min(w, h)
    long_side = max(w, h)
    aspect = short_side / long_side
    if not (0.4 < aspect < 1.0):
        return False

    # Extent: Ratio of contour area to bounding box area
    # A perfect rectangle is 1.0. Cards should be > 0.7.
    extent = float(area) / (w * h)
    if extent < 0.7:
        return False

    return True


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Orders coordinates as [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _is_duplicate_center(new_approx: np.ndarray, existing_candidates: List[np.ndarray], threshold: int = 20) -> bool:
    """Checks if a new contour center is too close to any existing candidate centers."""
    if not existing_candidates:
        return False
    new_center = np.mean(new_approx.reshape(4, 2), axis=0)
    for existing in existing_candidates:
        existing_center = np.mean(existing.reshape(4, 2), axis=0)
        if np.linalg.norm(new_center - existing_center) < threshold:
            return True
    return False


def get_all_candidates(image: np.ndarray) -> List[Dict[str, Any]]:
    """Finds all potential card-shaped quadrilaterals and returns their metadata."""
    if image is None:
        return []

    total_area = image.shape[0] * image.shape[1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    candidates = []
    for thresh_val in range(245, 30, -5):
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4 and _is_sane_quad(approx, total_area):
                if not _is_duplicate_center(approx, candidates):
                    candidates.append(approx)

    results = []
    for quad in candidates:
        pts = _order_points(quad.reshape(4, 2))
        M = cv2.getPerspectiveTransform(pts, np.array(
            [[0, 0], [200, 0], [200, 300], [0, 300]], dtype="float32"))
        crop_gray = cv2.warpPerspective(gray, M, (200, 300))
        crop_rgb = cv2.warpPerspective(rgb, M, (200, 300))
        _, _, w, h = cv2.boundingRect(quad)

        results.append({
            "coords": pts.tolist(),
            "raw_quad": quad,
            "area": float(cv2.contourArea(quad)),
            "relative_area": float(cv2.contourArea(quad) / total_area),
            "edge_density": score_edge_density(crop_gray),
            "color_variance": score_color_variance(crop_rgb),
            "aspect_fit": score_aspect_ratio_fit(w, h)
        })
    return results
