import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any


def filter_picture_cards(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Isolates Picture Cards using relative penalties rather than hard thresholds.
    Works across different camera scales and lighting.
    """
    if len(candidates) < 3:
        # If there's almost nothing, we can't cluster.
        # We just return everything to avoid losing data.
        return candidates

    # 1. Feature Extraction
    # We use features that stay stable even if the camera moves
    areas = np.array([c["area"] for c in candidates])
    densities = np.array([c["edge_density"] for c in candidates])
    aspects = np.array([c["aspect_fit"] for c in candidates])
    variances = np.array([c["color_variance"] for c in candidates])

    # 2. Normalize
    features = np.column_stack((areas, densities, aspects, variances))
    scaler = StandardScaler()
    norm_features = scaler.fit_transform(features)

    # 3. Cluster into 3 groups
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42,
                    n_init=10).fit(norm_features)
    labels = kmeans.labels_

    cluster_evals = []
    for i in range(n_clusters):
        mask = (labels == i)
        if not np.any(mask):
            cluster_evals.append({"id": i, "score": -1})
            continue

        count = np.sum(mask)
        avg_density = np.mean(densities[mask])
        avg_aspect = np.mean(aspects[mask])

        # BASE SCORE: Start with Density * Aspect Fit
        # This naturally favors "Artistic Rectangles"
        score = avg_density * avg_aspect

        # PENALTY 1: The "Lone Wolf" Penalty
        # If a cluster only has 1 item, it's likely an outlier.
        # We slash its score by 50%.
        if count == 1:
            score *= 0.1

        cluster_evals.append({"id": i, "score": score, "density": avg_density})

    # PENALTY 2: The "Numbered Card" Filter
    # In any frame, the numbered cards are the ones with the LEAST edge density.
    # We find the cluster with the lowest average density and tank its score.
    # This works regardless of what the actual density number is.
    min_density_cluster = min(
        cluster_evals, key=lambda x: x['density'] if x['score'] >= 0 else float('inf'))
    for eval_item in cluster_evals:
        if eval_item['id'] == min_density_cluster['id']:
            # Heavily penalize the 'quietest' cluster
            eval_item['score'] *= 0.1

    # 4. Selection
    winner_id = max(cluster_evals, key=lambda x: x['score'])['id']

    return [c for i, c in enumerate(candidates) if labels[i] == winner_id]
