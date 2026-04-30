import cv2
import numpy as np
from typing import List


def compute_phash(image: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """
    Computes a Perceptual Hash (pHash) to create a unique fingerprint of the card.

    Args:
        image: The cropped card image.
        hash_size: The square dimension of the hash (8x8 = 64 bits).
    """
    # 1. Pre-process: Grayscale and normalize lighting
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 2. Resize and Blur to remove high-frequency noise/grain
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)

    # 3. Compute Discrete Cosine Transform (DCT)
    dct = cv2.dct(np.float32(blurred))

    # 4. Extract low-frequency components (top-left)
    # We ignore the very first value (DC component) as it represents average color
    dct_low_freq = dct[:hash_size, :hash_size]
    med = np.median(dct_low_freq.flatten()[1:])

    # 5. Create a bit-mask (the fingerprint)
    return (dct_low_freq > med).flatten()


def get_hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
    """
    Counts how many bits differ between two hashes.
    A distance of 0 means identical images.
    """
    return np.count_nonzero(hash1 != hash2)


def is_duplicate(current_hash: np.ndarray, seen_hashes: List[np.ndarray], threshold: int = 5) -> bool:
    """
    Compares a new hash against a list of already saved hashes.
    """
    for existing_hash in seen_hashes:
        if get_hamming_distance(current_hash, existing_hash) <= threshold:
            return True
    return False


def get_unique(all, threshold=20):
    unique = []
    seen = []
    for img in all:
        hash = compute_phash(img)
        if is_duplicate(hash, seen, threshold):
            continue
        seen.append(hash)
        unique.append(img)
    return unique
