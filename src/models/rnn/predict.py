import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


def predict_recursive(
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

def predict_seq2seq(
    model: nn.Module,
    df_history: pd.DataFrame,
    df_test: pd.DataFrame,
    history_len: int,
    horizon: int,
    feature_cols: list[str],
    target_cols: list[str],
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    device='cpu',
) -> pd.DataFrame:
    model.eval()

    df_history = df_history.copy()
    preds = []

    for start in range(0, len(df_test), horizon):
        test_chunk = df_test.iloc[start : start + horizon]

        history_tail = df_history.tail(history_len)
        history_feature_scaled = feature_scaler.transform(history_tail[feature_cols])
        history_target_scaled = target_scaler.transform(history_tail[target_cols])

        encoder_x = np.hstack([history_feature_scaled, history_target_scaled])

        decoder_feature = feature_scaler.transform(test_chunk[feature_cols])

        encoder_x = torch.tensor(
            encoder_x,
            dtype=torch.float32,
        ).unsqueeze(0).to(device)

        decoder_feature = torch.tensor(
            decoder_feature,
            dtype=torch.float32,
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_scaled = model(
                encoder_x=encoder_x,
                decoder_feature=decoder_feature,
                decoder_targets=None,
                teacher_forcing_ratio=0.0,
            )

        pred_scaled = pred_scaled.cpu().numpy()[0]
        pred = target_scaler.inverse_transform(pred_scaled)

        pred_df = test_chunk.copy()
        pred_df[target_cols] = pred

        preds.append(pred_df[target_cols])

        df_history = pd.concat([df_history, pred_df[feature_cols + target_cols]])

    return pd.concat(preds)