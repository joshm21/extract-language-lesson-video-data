import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from sklearn.cluster import KMeans


# --- 1. SCORING FUNCTIONS ---

def score_edge_density(gray_crop):
    """Higher density usually means a picture card (complex art)."""
    edges = cv2.Canny(gray_crop, 50, 150)
    density = np.sum(edges > 0) / \
        float(gray_crop.shape[0] * gray_crop.shape[1])
    return float(density)


def score_color_variance(rgb_crop):
    """Picture cards have high color variety; numbered cards are mostly white/red/black."""
    # Calculate standard deviation of color channels
    std_dev = np.mean(np.std(rgb_crop, axis=(0, 1)))
    return float(std_dev)


def score_aspect_ratio_fit(w, h):
    """How close is it to a standard 2.5 x 3.5 card ratio (approx 0.71)?"""
    target = 0.714
    current = float(w) / h
    if current > 1:
        current = 1/current  # Handle sideways cards
    # Score is 1.0 for perfect match, drops as it deviates
    return float(1.0 - abs(target - current))


def score_relative_area(contour_area, total_img_area):
    """Calculates what percentage of the image the card occupies."""
    # Returns a value between 0.0 and 1.0
    return float(contour_area / total_img_area)

# --- 2. GEOMETRY UTILITIES ---


def is_sane_quad(approx):
    if not cv2.isContourConvex(approx):
        return False
    x, y, w, h = cv2.boundingRect(approx)
    aspect_ratio = float(w) / h
    if aspect_ratio < 0.2 or aspect_ratio > 5.0:
        return False
    area = cv2.contourArea(approx)
    if area < (w * h * 0.6):
        return False
    return True


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# --- 3. THE PIPELINE ---

def get_dynamic_winners(all_card_data):
    # 1. Strip out the gargantuan background shapes first
    candidates = [c for c in all_card_data if c["relative_area"] < 0.10]

    if len(candidates) < 2:
        return candidates

    # 2. Extract our two best features: Area and Edge Density
    areas = np.array([c["area"] for c in candidates])
    densities = np.array([c["edge_density"] for c in candidates])

    # Normalize the data so Area (in the thousands) doesn't overpower Density (decimals)
    norm_areas = (areas - areas.min()) / (areas.max() - areas.min())
    norm_densities = (densities - densities.min()) / (densities.max() - densities.min())

    # Create a 2D array of our features
    features = np.column_stack((norm_areas, norm_densities))

    # 3. Apply K-Means Clustering to group into exactly 2 clusters (Numbers vs Pictures)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(features)
    labels = kmeans.labels_

    # 4. Identify which cluster represents the "Picture Cards"
    # We know picture cards are physically larger on average.
    cluster_0_avg_area = np.mean(areas[labels == 0])
    cluster_1_avg_area = np.mean(areas[labels == 1])

    picture_cluster_id = 0 if cluster_0_avg_area > cluster_1_avg_area else 1

    # 5. Extract the winners based on the winning cluster ID
    winners = [c for i, c in enumerate(candidates) if labels[i] == picture_cluster_id]

    return winners


def run_pipeline(image_path):
    image = cv2.imread(image_path)
    total_img_area = image.shape[0] * image.shape[1]
    if image is None:
        return

    orig_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # A. CANDIDATE SWEEP
    candidates = []
    for thresh_val in range(240, 40, -10):
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (image.shape[0] * image.shape[1] * 0.005):
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

                if len(approx) == 4 and is_sane_quad(approx):
                    # Duplicate check using center distance
                    c1 = np.mean(approx.reshape(4, 2), axis=0)
                    if not any(np.linalg.norm(c1 - np.mean(c.reshape(4, 2), axis=0)) < 50 for c in candidates):
                        candidates.append(approx)

    # B. SCORING & DATA COLLECTION
    all_card_data = []
    for quad in candidates:
        pts = order_points(quad.reshape(4, 2))
        dst = np.array([[0, 0], [200, 0], [200, 300],
                       [0, 300]], dtype="float32")
        M = cv2.getPerspectiveTransform(pts, dst)

        crop_gray = cv2.warpPerspective(gray, M, (200, 300))
        crop_rgb = cv2.warpPerspective(orig_rgb, M, (200, 300))
        _, _, w, h = cv2.boundingRect(quad)

        # Build the score dictionary
        stats = {
            "coords": pts.tolist(),
            "edge_density": score_edge_density(crop_gray),
            "color_variance": score_color_variance(crop_rgb),
            "aspect_fit": score_aspect_ratio_fit(w, h),
            "area": float(cv2.contourArea(quad)),
            "relative_area": score_relative_area(cv2.contourArea(quad), total_img_area),
            "raw_quad": quad  # kept for drawing
        }
        all_card_data.append(stats)

    # C. FILTERING LOGIC (TEST COMBINATIONS HERE)
#    winners = []
#    for card in all_card_data:
#        # 1. KILL THE BACKGROUND: Must be less than 10% of image
#        not_background = card["relative_area"] < 0.10
#
#        # 2. KILL THE NUMBER CARDS: Numbers max out at ~2650 area.
#        is_card_sized = card["area"] > 3500
#
#        # 3. IDENTIFY ART: Numbers peak at 0.014 density. Art starts at 0.024.
#        has_high_detail = card["edge_density"] > 0.020
#
#        # Note: color_variance is completely removed!
#
#        if not_background and is_card_sized and has_high_detail:
#            winners.append(card)

    # C. OR USE K-CLUSTERS TO DYNAMICALLY SET THRESHOLDS
    winners = get_dynamic_winners(all_card_data)

    # D. SAVE TO JSON
    # Remove 'raw_quad' before saving as it isn't JSON serializable
    json_data = [{k: v for k, v in c.items() if k != 'raw_quad'}
                 for c in all_card_data]
    with open('card_analysis.json', 'w') as f:
        json.dump(json_data, f, indent=4)

    # E. VISUALIZATION
    plt.figure(figsize=(12, 8))
    display_img = orig_rgb.copy()

    # Draw all candidates (Light Yellow)
    for card in all_card_data:
        cv2.polylines(display_img, [card["raw_quad"]],
                      True, (255, 255, 200), 2)

    # Draw winners (Medium Green)
    for card in winners:
        cv2.polylines(display_img, [card["raw_quad"]], True, (50, 205, 50), 6)

    plt.imshow(display_img)
    plt.title(
        f"Candidates: {len(all_card_data)} (Yellow) | Filtered Winners: {len(winners)} (Green)")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    run_pipeline(args.filename)
