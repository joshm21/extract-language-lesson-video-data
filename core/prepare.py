import cv2
import numpy as np


def get_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts a BGR image to single-channel grayscale. 
    Reduces data complexity for contour detection.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Smoothes the image to reduce high-frequency noise like wood grain.

    Args:
        image: The input image array.
        ksize: The blur radius. Larger kernels average more pixels, 
               removing more grain but rounding sharp corners.
    """
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def apply_dilate(binary_image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Expands white regions. 

    Args:
        kernel_size: The 'neighborhood' of expansion. Larger kernels 
                     bridge bigger gaps in card outlines.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(binary_image, kernel, iterations=iterations)


def apply_erode(binary_image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Shrinks white regions. 

    Args:
        kernel_size: Larger kernels remove larger noise islands but 
                     risk thinning card borders into non-existence.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(binary_image, kernel, iterations=iterations)


def apply_open(binary_image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Erosion followed by Dilation. 
    Kernel size determines the maximum size of 'dots' to be deleted.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)


def apply_close(binary_image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Dilation followed by Erosion. 
    Kernel size determines the maximum 'gap' distance that can be healed.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
