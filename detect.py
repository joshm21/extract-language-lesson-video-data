import cv2
import numpy as np
from typing import List

def detect_basic(image, thresh_val=127):
    """A basic detector. Challenge: Can you make this check multiple thresholds?"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    quads = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            quads.append(approx)
    return quads


def get_threshold_frame(gray: np.ndarray, thresh_val: int) -> np.ndarray:
    """
    Simple binary threshold. Useful for students to visualize 
    what the computer 'sees' before contouring.
    """
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    return thresh


def find_quadrilaterals(
    binary_image: np.ndarray,
    approx_epsilon: float = 0.02
) -> List[np.ndarray]:
    """
    Finds 4-sided polygons in a binary image.

    Args:
        binary_image: A black and white (thresholded) image.
        approx_epsilon: Fine-tuning for how 'perfect' the rectangle must be.
                        Lower values allow for more curved/imperfect edges.
    """
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    quads = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, approx_epsilon * peri, True)

        # Only keep shapes with exactly 4 corners
        if len(approx) == 4:
            quads.append(approx)
    return quads


def draw_detections(image: np.ndarray, quads: List[np.ndarray]) -> np.ndarray:
    """
    Draws all found quadrilaterals onto a copy of the image for debugging.
    """
    output = image.copy()
    cv2.drawContours(output, quads, -1, (0, 255, 0), 2)
    return output


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders coordinates consistently: [top-left, top-right, bottom-right, bottom-left].
    Essential for consistent cropping and deskewing.
    """
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_cards_single_threshold(image: np.ndarray, threshold: int = 127):
    """Simple detection. Students will notice this fails in varied lighting."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return find_quadrilaterals(thresh)


def sweep_detect_cards(
    image: np.ndarray,
    min_thresh: int = 30,
    max_thresh: int = 245,
    step: int = 10
) -> List[np.ndarray]:
    """
    The 'Sweep' idea: Iterates through different lighting thresholds to find cards.
    This helps find cards in both shadows and highlights.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    unique_quads = []

    # Iterate through threshold values
    for val in range(max_thresh, min_thresh, -step):
        thresh = get_threshold_frame(blurred, val)
        candidates = find_quadrilaterals(thresh)

        for quad in candidates:
            # Workshop challenge: students can add logic here to
            # filter out duplicates based on center-point distance
            unique_quads.append(quad)

    return unique_quads
