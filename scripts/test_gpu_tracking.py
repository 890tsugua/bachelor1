
# Load a csv
import pandas as pd
import os
import numpy as np
import torch

def load_csv(file_path):
    """
    Load a CSV file and return a DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    return pd.read_csv(file_path)

# Method to extract x and y coordinates from the DataFrame
def extract_predictions_from_gpu_tracking(df):
    """
    Extract x and y coordinates from the DataFrame.
    Assumes the DataFrame has 'x' and 'y' columns.
    x and y coordinates with the same "frame" id all go into the same dictionary as tensor (N,2) under key 'box_centers'.
    Returns a list of dictionaries. Each dictionary contains 'box_centers' key with a tensor of shape (N, 2).
    """
    if 'x' not in df.columns or 'y' not in df.columns:
        raise ValueError("DataFrame must contain 'x' and 'y' columns")
    
    # Remove all other columns except 'frame', 'x', and 'y'
    df = df[['frame', 'x', 'y']].dropna()

    # Group by 'frame' and extract coordinates
    grouped = df.groupby('frame')
    predictions = []
    for frame, group in grouped:
        coords = group[['x', 'y']].values
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        predictions.append({'frame': frame, 'box_centers': coords_tensor})
    return predictions

def dist_evaluate_prediction(prediction, target, dist_thresh = 0.5):
    """
    Evaluate a single image prediction against the target. 
    Prediction and target are dictionaries containing keys...
    """
    pred_coords = prediction['box_centers'] # Tensor (N,2)
    gt_coords= target['positions']  # Tensor (M,2)

    # Match each prediction to at most one ground truth using greedy matching
    distance_matrix = torch.cdist(pred_coords.unsqueeze(0), gt_coords.unsqueeze(0))  # Calculate pairwise distances
    distance_matrix = distance_matrix.squeeze(0)  # Remove the extra dimension

    matches = []
    used_preds = set()
    used_gts = set()

    pairs = [(i, j, distance_matrix[i,j].item())
             for i in range(distance_matrix.shape[0])
             for j in range(distance_matrix.shape[1])] # Makes a list of tuples. All possible pairs

    # Sort pairs by distance in descending order
    pairs.sort(key=lambda x: x[2], reverse=True)

    # Now greedy matching.
    for i, j, distance in pairs:
        if distance < dist_thresh:
            continue
        if i not in used_preds and j not in used_gts:
            matches.append((i,j))
            used_preds.add(i)
            used_gts.add(j)

    # Calculate TP, FP, FN
    tp = len(matches)
    fp = pred_coords.shape[0] - tp
    fn = gt_coords.shape[0] - tp

    ji = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2*(precision*recall))/(precision+recall) if (precision + recall) > 0 else 0

    # ADD LOCALIZATION ERROR
    loc_errors = []
    square_loc_errors = []
    for (i,j) in matches:
        dx = target['positions'][j][0] - prediction['box_centers'][i][0]
        dy = target['positions'][j][1] - prediction['box_centers'][i][1]
        d2 = (dx**2 + dy**2)
        d2 = d2.item() if isinstance(d2, torch.Tensor) else d2
        square_loc_errors.append(d2)
        loc_errors.append(d2**0.5)
    
    mse = np.mean(square_loc_errors)
    stdse = np.std(square_loc_errors)
    me = np.mean(loc_errors)
    stde = np.std(loc_errors)

    return precision, recall, f1, ji, me, stde, mse, stdse

def dist_evaluate_predictions(predictions, targets, dist_thresh = 0.5):
    """
    A method that is able to evaluate a batch (list) of predictions.
    Inputs should be lists of dictionaries, where each dictionary contains:
    - Predictions should contain 'box_centers': Tensor of shape (N, 2) with predicted centers
    - Ground truths should contain 'positions': Tensor of shape (N, 2)
    """

    precisions = []
    recalls = []
    f1s = []
    jis = []
    mean_errors = []
    std_errors = []
    mses = []
    stdses = []

    for prediction, target in zip(predictions, targets):
        p, r, f1, ji, mean_error, std_error, mse, stdse = dist_evaluate_prediction(prediction, target, dist_thresh)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        jis.append(ji)
        mean_errors.append(mean_error)
        std_errors.append(std_error)
        mses.append(mse)
        stdses.append(stdse)

    precision = np.mean(precisions)
    recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)
    avg_ji = np.mean(jis)
    avg_me = np.mean(mean_errors)
    avg_stde = np.mean(std_errors)
    avg_mse = np.mean(mses)
    avg_stdse = np.mean(stdses)

    std_precision = np.std(precisions)
    std_recall = np.std(recalls)
    std_f1 = np.std(f1s)
    std_ji = np.std(jis)
    std_me = np.std(mean_errors)
    std_mse = np.std(mses)

    return {"mean_precision": precision, "mean_recall": recall, "mean_f1": avg_f1, "mean_ji": avg_ji, 
            "mean_loc_error": avg_me, "avg_std_loc_error": avg_stde, "mean_mean_squared_error": avg_mse, "avg_std_mean_squared_error": avg_stdse,
            "std_precision": std_precision, "std_recall": std_recall, "std_f1": std_f1, "std_ji": std_ji, 
            "std_loc_error": std_me, "std_mean_squared_error": std_mse}

