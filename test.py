import cv2
import numpy as np


def test_blur_sweep(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # We sweep BLUR kernels instead of THRESHOLD values
    # 3 = sharp (for small numbers), 13 = heavy (for textured wood)
    all_found_quads = []

    for k in [3, 7, 13]:
        blurred = cv2.GaussianBlur(gray, (k, k), 0)

        # Use Otsu to find the "best" high threshold for this specific blur level
        high_thresh, _ = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edged = cv2.Canny(blurred, high_thresh * 0.5, high_thresh)

        # Close edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edged, kernel, iterations=1)

        # Find quads (reusing your area filter of 1000)
        cnts, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_scale_quads = []
        for c in cnts:
            if cv2.contourArea(c) > 1000:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    current_scale_quads.append(approx)
                    all_found_quads.append(approx)

        # Save a debug image for this specific blur level
        res = image.copy()
        cv2.drawContours(res, current_scale_quads, -1, (0, 255, 0), 3)
        cv2.imwrite(f"blur_sweep_k{k}.jpg", res)
        print(f"Blur K={k}: Found {len(current_scale_quads)} quads.")

    print(f"Total quads to pass to pHash: {len(all_found_quads)}")


if __name__ == "__main__":
    test_blur_sweep("sample.jpg")
