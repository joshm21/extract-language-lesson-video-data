import cv2
import numpy as np
from typing import Dict

# --- SECTION 1: PRE-PROCESSING ---


def to_grayscale(state: Dict) -> Dict:
    """
    Converts the current image to single-channel grayscale.

    Args:
        state: Dictionary containing 'current_image' (BGR or Gray).

    Effect: Reduces data complexity. If the image is already grayscale, 
            it passes through unchanged to prevent errors.
    """
    img = state["current_image"]
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return {"current_image": img, "auto_save": img}


def to_blurred(state: Dict, ksize: int = 5) -> Dict:
    """
    Smoothes the current image to reduce high-frequency noise.

    Args:
        state: Dictionary containing 'current_image'.
        ksize: Blur radius (must be odd). Higher values average more pixels, 
               hiding more wood grain but rounding sharp card corners.
    """
    img = state["current_image"]
    if ksize % 2 == 0:
        ksize += 1

    res = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return {"current_image": res, "auto_save": res}


# --- SECTION 2: BINARIZATION ---

def at_global_threshold(state: Dict, threshold: int = 127) -> Dict:
    """
    Converts the current image to binary using a single fixed cutoff.

    Args:
        state: Dictionary containing 'current_image'.
        threshold: Brightness cutoff (0-255). Higher values are more restrictive, 
                   turning more of the image into black.
    """
    img = state["current_image"]
    _, res = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return {"current_image": res, "auto_save": res}


def at_adaptive_threshold(state: Dict, block_size: int = 11, c_val: int = 2) -> Dict:
    """
    Converts to binary using local neighborhood brightness.

    Args:
        state: Dictionary containing 'current_image'.
        block_size: Size of local window (odd). Larger = more context.
        c_val: Sensitivity constant. Higher = stricter thresholding.
    """
    img = state["current_image"]
    if block_size % 2 == 0:
        block_size += 1

    res = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c_val
    )
    return {"current_image": res, "auto_save": res}


def at_canny_edges(state: Dict, low: int = 50, high: int = 150) -> Dict:
    """
    Converts current image into a binary map of high-contrast edges.

    Args:
        state: Dictionary containing 'current_image'.
        low: Minimum edge strength.
        high: Strength required to start a line. Higher values ensure only 
              the clearest card borders are detected.
    """
    img = state["current_image"]
    res = cv2.Canny(img, low, high)
    return {"current_image": res, "auto_save": res}


# --- SECTION 3: MORPHOLOGY ---

def do_dilation(state: Dict, kernel_size: int = 3, iterations: int = 1) -> Dict:
    """
    Expands white regions in the current binary image.

    Args:
        state: Dictionary containing 'current_image'.
        kernel_size: Neighborhood of expansion. Larger kernels bridge bigger 
                     gaps in detected card outlines.
    """
    img = state["current_image"]
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    res = cv2.dilate(img, kernel, iterations=iterations)
    return {"current_image": res, "auto_save": res}


def do_erosion(state: Dict, kernel_size: int = 3, iterations: int = 1) -> Dict:
    """
    Shrinks white regions in the current binary image.

    Args:
        state: Dictionary containing 'current_image'.
        kernel_size: Higher values remove larger noise islands but risk 
                     thinning card borders into non-existence.
    """
    img = state["current_image"]
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    res = cv2.erode(img, kernel, iterations=iterations)
    return {"current_image": res, "auto_save": res}


def do_opening(state: Dict, kernel_size: int = 3) -> Dict:
    """
    Erosion followed by Dilation: removes small white noise dots.

    Args:
        state: Dictionary containing 'current_image'.
        kernel_size: Maximum size of 'speckles' to be deleted.
    """
    img = state["current_image"]
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    res = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return {"current_image": res, "auto_save": res}


def do_closing(state: Dict, kernel_size: int = 3) -> Dict:
    """
    Dilation followed by Erosion: heals small gaps in lines.

    Args:
        state: Dictionary containing 'current_image'.
        kernel_size: Maximum gap distance that can be filled.
    """
    img = state["current_image"]
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    res = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return {"current_image": res, "auto_save": res}
