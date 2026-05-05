import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Optional, Any
from . import visualize as viz
from core.score import props


def k_means(
    state: Dict[str, Any],
    k: int = 2,
    features: List[str] = [props.area],
    select_max: Optional[str] = None,
    select_min: Optional[str] = None
) -> Dict[str, Any]:
    """
    Groups quads into K clusters using unsupervised learning based on scoring metrics, then selects a 'winning' cluster.

    This allows users to see how unsupervised learning can distinguish between 
    'Number Cards' and 'Picture Cards' without manual thresholds.

    Args:
        state: The current pipeline state containing 'quads', 'scores', and 'raw_frame'.
        k: Number of clusters to find.
        features: List of metrics from score.props to use as dimensions for clustering.
        select_max: The metric used to identify the target cluster (highest average wins).
        select_min: The metric used to identify the target cluster (lowest average wins).
    """
    all_quads = state.get("quads", [])
    scores = state.get("scores", [])
    raw_frame = state.get("raw_frame")

    if not all_quads or not scores:
        return {"passed_indices": []}

    # 1. Prepare Feature Matrix
    # We extract the requested metrics for every quad into a 2D array
    feature_data = []
    for s in scores:
        feature_data.append([s.get(f, 0.0) for f in features])

    X = np.array(feature_data, dtype=np.float32)

    # 2. Normalize Features (Min-Max Scaling)
    # K-Means is distance-based, so we must scale features to a [0, 1] range:
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    # Prevent division by zero if a feature has no variance
    X_norm = (X - x_min) / (x_max - x_min + 1e-6)

    # 3. Perform Clustering
    # Ensure k never exceeds the number of available samples
    effective_k = min(k, len(X_norm))

    model = KMeans(n_clusters=effective_k, n_init=10, random_state=42)
    labels = model.fit_predict(X_norm)

    # 4. Determine Target Cluster
    # We calculate the mean value of the 'selection metric' for each cluster
    target_metric = select_max or select_min or features[0]
    cluster_means = []
    for i in range(k):
        cluster_values = [scores[j][target_metric]
                          for j in range(len(labels)) if labels[j] == i]
        cluster_means.append(np.mean(cluster_values)
                             if cluster_values else 0.0)

    target_cluster_idx = np.argmax(
        cluster_means) if select_max else np.argmin(cluster_means)
    passed_indices = [i for i, label in enumerate(
        labels) if label == target_cluster_idx]

    # 5. Visualization
    # We generate a distinct color for each cluster, forcing the target to Green.
    viz_groups = []
    for cluster_id in range(k):
        if cluster_id == target_cluster_idx:
            color = (0, 255, 0)  # Green for the "Winner"
        else:
            # Clamps at 0 so it never goes negative
            # Also varies the Blue and Red channels slightly for high k
            r = max(0, 255 - (cluster_id * 40))
            b = (cluster_id * 50) % 256
            color = (b, 0, r)

        indices = [i for i, label in enumerate(labels) if label == cluster_id]
        viz_groups.append({
            'quads': [all_quads[i] for i in indices],
            'labels': [f"Cluster {cluster_id}: {i}" for i in indices],
            'color': color
        })

    viz_image = viz.draw_multiple_quad_groups(raw_frame, viz_groups)

    return {
        "passed_indices": passed_indices,
        "auto_save": viz_image
    }
