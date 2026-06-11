import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


def create_seq2seq_eval_loader(
    df_eval: pd.DataFrame,
    df_start: pd.DataFrame,
    history_len: int,
    horizon: int,
    feature_cols: list[str],
    target_cols: list[int],
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    batch_size=64
) -> DataLoader:
    df_eval_full = pd.concat([df_start.tail(history_len), df_eval])

    feature_scaled = feature_scaler.transform(df_eval_full[feature_cols])
    target_scaled = target_scaler.transform(df_eval_full[target_cols])

    data_scaled = np.hstack([feature_scaled, target_scaled])

    X_enc, X_dec, y = create_seq2seq_windows(
        data=data_scaled,
        history_len=history_len,
        horizon=horizon,
        target_start_index=len(feature_cols),
    )

    dataset = TensorDataset(X_enc, X_dec, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return loader

def create_seq2seq_train_loader(
    df: pd.DataFrame,
    history_len: int,
    horizon: int,
    feature_cols: list[str],
    target_cols: list[int],
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    batch_size=64
) -> DataLoader:
    feature_scaled = feature_scaler.fit_transform(df[feature_cols])
    target_scaled = target_scaler.fit_transform(df[target_cols])

    data_scaled = np.hstack((feature_scaled, target_scaled))
    X_enc, X_dec, y = create_seq2seq_windows(
        data=data_scaled,
        history_len=history_len,
        horizon=horizon,
        target_start_index=len(feature_cols)
    )
    
    dataset = TensorDataset(X_enc, X_dec, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader
    
def create_seq2seq_windows(
    data: np.ndarray | pd.DataFrame,
    history_len: int,
    horizon: int,
    feature_cols=None,
    target_cols=None,
    target_start_index=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(data, np.ndarray):
        if target_start_index is None:
            raise ValueError('Need target_start_index')

        array = data.astype('float32', copy=False)
        encoder_data = array
        decoder_feature_data = array[:, :target_start_index]
        target_data = array[:, target_start_index:]
    else:
        if feature_cols is None or target_cols is None:
            raise ValueError('Need feature_cols and target_cols')

        encoder_cols = feature_cols + target_cols
        encoder_data = data[encoder_cols].values.astype('float32')
        decoder_feature_data = data[feature_cols].values.astype('float32')
        target_data = data[target_cols].values.astype('float32')

    X_encoder = []
    X_decoder_feature = []
    y = []

    max_start = len(encoder_data) - history_len - horizon + 1

    for start in range(max_start):
        history_start = start
        history_end = start + history_len

        forecast_start = history_end
        forecast_end = forecast_start + horizon

        X_encoder.append(encoder_data[history_start:history_end])
        X_decoder_feature.append(decoder_feature_data[forecast_start:forecast_end])
        y.append(target_data[forecast_start:forecast_end])

    return (
        torch.tensor(np.array(X_encoder), dtype=torch.float32),
        torch.tensor(np.array(X_decoder_feature), dtype=torch.float32),
        torch.tensor(np.array(y), dtype=torch.float32),
    )

def create_sequences(data, target_start_index, lags_length):
    X, y = [], []
    for i in range(len(data) - lags_length):
        X.append(data[i : i + lags_length])
        y.append(data[i + lags_length, target_start_index:])
    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        torch.tensor(np.array(y), dtype=torch.float32)
    )