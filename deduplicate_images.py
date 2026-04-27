import cv2
import numpy as np
from typing import List


def compute_phash(image: np.ndarray, hash_size: int = 16, highfreq_factor: int = 4) -> np.ndarray:
    """
    Computes a 64-bit Perceptual Hash (pHash) of an image using Discrete Cosine Transform.

    Args:
        image: The source image array (BGR).
        hash_size: The size of the resulting hash (8x8 = 64 bits).
        highfreq_factor: The multiplier for the initial resize to include more detail before DCT.

    Returns:
        A 1D boolean numpy array representing the image fingerprint.
    """
    # 1. Grayscale and resize to a small square
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # NEW: Normalize lighting so dark/bright versions of the same card look identical
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # A 5x5 kernel helps remove high-frequency noise (glare/grain)
    # that causes unnecessary bit-flips in the pHash.
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    img_size = hash_size * highfreq_factor
    resized = cv2.resize(gray, (img_size, img_size),
                         interpolation=cv2.INTER_AREA)

    # 2. Compute the Discrete Cosine Transform (DCT)
    # DCT separates the image into high and low frequency components
    dct = cv2.dct(np.float32(resized))

    # 3. Extract the top-left low-frequency components
    dct_low_freq = dct[:hash_size, :hash_size]

    # 4. Compute the median and create a bit-mask
    # We exclude the first DC component (0,0) for better luminance invariance
    med = np.median(dct_low_freq.flatten()[1:])
    diff = dct_low_freq > med

    return diff.flatten()


def get_hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
    """
    Calculates the Hamming distance between two pHashes.

    Args:
        hash1: First boolean hash array.
        hash2: Second boolean hash array.

    Returns:
        The number of differing bits. A lower number means higher similarity.
    """
    return np.count_nonzero(hash1 != hash2)


def deduplicate_images(images: List[np.ndarray], threshold: int = 100) -> List[np.ndarray]:
    """
    Takes a list of images and returns only those that are perceptually unique.

    Args:
        images: A list of image arrays (numpy).
        threshold: The maximum Hamming distance to consider two images as duplicates.
                   (0-2 is very strict, 5-10 is standard for near-duplicates).

    Returns:
        A list of unique image arrays.
    """
    unique_images: List[np.ndarray] = []
    seen_hashes: List[np.ndarray] = []

    for img in images:
        if img is None:
            continue

        current_hash = compute_phash(img)

        # Check if this hash is similar to any we've already kept
        is_duplicate = False
        for existing_hash in seen_hashes:
            if get_hamming_distance(current_hash, existing_hash) <= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_images.append(img)
            seen_hashes.append(current_hash)

    return unique_images
