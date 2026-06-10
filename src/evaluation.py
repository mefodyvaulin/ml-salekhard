import numpy as np
import pandas as pd
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mean_error = np.mean(y_pred - y_true)
    nse = nash_sutcliffe_efficiency(y_true, y_pred)
    
    eval_dict = {
        'RMSE': rmse,
        'MAE': mae,
        'ME': mean_error,
        'NSE': nse
    }
    
    return eval_dict


def evaluate_by_depth(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    depth_names: list[str],
) -> pd.DataFrame:
    if y_true.shape != y_pred.shape:
        raise ValueError(f'y_true and y_pred must have the same shape')

    n_depths = y_true.shape[1]

    if len(depth_names) != n_depths:
        raise ValueError(f'depth_names must contain {n_depths} names')

    rows = []
    for i, depth_name in enumerate(depth_names):
        depth_true = y_true[:, i]
        depth_pred = y_pred[:, i]

        rows.append({
            'depth': depth_name,
            'RMSE': root_mean_squared_error(depth_true, depth_pred),
            'MAE': mean_absolute_error(depth_true, depth_pred),
            'ME': np.mean(depth_pred - depth_true),
            'NSE': nash_sutcliffe_efficiency(depth_true, depth_pred),
        })
        
    df = pd.DataFrame(rows)
    df.index = df['depth']

    return df


def nash_sutcliffe_efficiency(y_true: np.ndarray, y_pred: np.ndarray):
    return r2_score(y_true, y_pred)
