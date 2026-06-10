import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


def predict(model: nn.Module, start_lags, future_features, device='cpu'):
    model.eval()
    current_lags = start_lags.clone().to(device)
    future_features = future_features.to(device)
    y_pred = []
    
    with torch.no_grad():
        for i in range(len(future_features)):     
            lags_tensor = current_lags.unsqueeze(0)
            pred = model(lags_tensor)
            y_pred.append(pred.cpu().numpy()[0])
            
            next_step_features = future_features[i]
            next_step_vector = torch.cat((next_step_features, pred[0]))
            current_lags = torch.vstack((current_lags[1:], next_step_vector))

    return np.array(y_pred)

def get_full_predict(
    model: nn.Module,
    df_test: pd.DataFrame,
    test_start: pd.DataFrame,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    features_cols: list[str],
    target_cols: list[str],
    device='cpu'
):
    start_features_scaled = feature_scaler.transform(test_start[features_cols])
    start_target_scaled = target_scaler.transform(test_start[target_cols])

    current_lags = np.hstack((start_features_scaled, start_target_scaled))
    future_features_scaled = feature_scaler.transform(df_test[features_cols])

    start_lags_tensor = torch.tensor(current_lags, dtype=torch.float32)
    future_features_tensor = torch.tensor(future_features_scaled, dtype=torch.float32)

    y_pred_scaled = predict(model, start_lags_tensor, future_features_tensor, device=device)

    y_pred = target_scaler.inverse_transform(y_pred_scaled)

    return y_pred