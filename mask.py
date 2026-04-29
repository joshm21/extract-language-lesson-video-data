import cv2
import numpy as np


def global_thresh(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    """
    Converts to binary image using a single fixed cutoff value.

    Args:
        threshold: The brightness cutoff (0-255). 
                   Higher = stricter (more pixels become black).
    """
    _, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return thresh


def adaptive(image: np.ndarray, block_size: int = 11, c_val: int = 2) -> np.ndarray:
    """
    Converts to binary image by looking at local pixel neighborhoods.
    Useful for handling shadows on the wood table.

    Args:
        block_size: Size of the local window (must be odd). 
                    Higher = looks at a larger area for context.
        c_val: Sensitivity constant. 
               Higher = stricter (more pixels become black).
    """
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c_val
    )


def canny(image: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """
    Converts to binary image by finding high-contrast edges.

    Args:
        low: Minimum edge strength.
        high: Strength required to 'start' a line. 
              Higher = only finds the sharpest card borders.
    """
    return cv2.Canny(image, low, high)
