import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from typing import Dict, List, Any
from . import visualize as viz
from .score import props


def knn(
    state: Dict[str, Any],
    train_csv: str,
    k: int = 5,  # Higher k is better for meaningful probabilities
    features: List[str] = [props.area, props.edge_density],
    target_label: int = 1,
    threshold: float = 0.6  # Only pass if confidence is > 60%
) -> Dict[str, Any]:
    """
    Identifies quads using KNN with a confidence threshold.

    Args:
        state: Pipeline state containing 'quads', 'scores', and 'raw_frame'.
        train_csv: CSV file with 'label' and feature columns created by students.
        k: Number of neighbors. If k=5, probabilities will be in 20% increments.
        features: Metrics from score.props to use for distance calculation.
        target_label: The class ID to extract (e.g., 1 for Picture Cards).
        threshold: Minimum probability (0.0 to 1.0) required to 'pass' a quad.
    """
    all_quads = state.get("quads", [])
    scores = state.get("scores", [])
    raw_frame = state.get("raw_frame")

    if not all_quads or not scores:
        return {"passed_indices": []}

    # 1. Load Training Data
    with open(train_csv, mode='r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    X_train = np.array([
        [float(row[feat]) for feat in features]
        for row in rows
    ], dtype=np.float32)
    y_train = np.array([int(row['label']) for row in rows])

    # 2. Prepare and Normalize Data
    # We normalize using training bounds to keep the feature space consistent.
    current_data = np.array([[s.get(f, 0.0)
                            for f in features] for s in scores])

    x_min, x_max = X_train.min(axis=0), X_train.max(axis=0)
    X_train_norm = (X_train - x_min) / (x_max - x_min + 1e-6)
    X_test_norm = (current_data - x_min) / (x_max - x_min + 1e-6)

    # 3. Predict Probabilities
    # Ensure k is at least 1 and never more than the number of training samples
    effective_k = max(1, min(k, len(X_train_norm)))
    model = KNeighborsClassifier(n_neighbors=effective_k)
    model.fit(X_train_norm, y_train)

    # Get probability for each class (returns a matrix: [samples, classes])
    probs = model.predict_proba(X_test_norm)
    class_list = model.classes_.tolist()

    # Find the column index for our target label
    if target_label in class_list:
        target_col_idx = class_list.index(target_label)
        # Extract probabilities for just our target class
        confidences = probs[:, target_col_idx]
    else:
        confidences = np.zeros(len(scores))

    # 4. Apply Threshold
    # Only pass indices where the model is confident enough
    passed_indices = [i for i, conf in enumerate(
        confidences) if conf >= threshold]

    # 5. Visualization with Confidence Labels
    viz_groups = []
    for i, conf in enumerate(confidences):
        label_id = model.predict(X_test_norm[i:i+1])[0]
        is_passed = i in passed_indices

        # Color: Green if passed, Yellow if correct class but low confidence, Red otherwise
        if is_passed:
            color = (0, 255, 0)
        elif label_id == target_label:
            color = (0, 255, 255)  # Yellow: "I think so, but I'm not sure"
        else:
            color = (0, 0, 200)   # Red: "Definitely not"

        viz_groups.append({
            'quads': [all_quads[i]],
            'labels': [f"ID:{label_id} ({conf*100:.0f}%)"],
            'color': color
        })

    viz_image = viz.draw_multiple_quad_groups(raw_frame, viz_groups)

    return {
        "passed_indices": passed_indices,
        "auto_save": viz_image
    }
