import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from utils import ensure_skforecast_compat, process_data, process_linear_regression


class BaseForecaster:
    depth_columns = []

    def predict(self, history_df, steps, future_exog=None):
        raise NotImplementedError


class XGBoostV7Wrapper(BaseForecaster):
    def __init__(self, path):
        self.model = joblib.load(path)
        for forecaster in self.model.forecasters.values():
            ensure_skforecast_compat(forecaster)
        self.target_cols = list(self.model.forecasters.keys())
        self.exog_cols = list(next(iter(self.model.forecasters.values())).exog_names_in_)
        self.depth_columns = self.target_cols
        self.max_lag = max(f.max_lag for f in self.model.forecasters.values())

    def predict(self, history_df, steps, future_exog):
        last_window = history_df[self.target_cols].tail(self.max_lag)
        forecast = self.model.predict(
            steps=steps,
            last_window=last_window,
            exog=future_exog[self.exog_cols],
        )
        forecast.index = future_exog.index
        return forecast[self.target_cols]


class LSTMRegressor(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.out = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.out(out[:, -1, :])


class LSTMV1Wrapper(BaseForecaster):
    def __init__(self, bundle_path):
        bundle = joblib.load(bundle_path)
        self.config = bundle["model_config"]
        self.model = LSTMRegressor(**self.config)
        self.model.load_state_dict(bundle["model_state"])
        self.model.eval()
        self.feat_scaler = bundle["feature_scaler"]
        self.target_scaler = bundle["target_scaler"]
        self.seq_len = bundle["seq_len"]
        self.target_cols = bundle["target_cols"]
        self.feat_cols = bundle["feature_cols"]
        self.depth_columns = self.target_cols

    def predict(self, history_df, steps, future_exog):
        current_window = history_df.tail(self.seq_len)
        preds = []
        last_window_feats = self.feat_scaler.transform(current_window[self.feat_cols])
        last_window_targets = self.target_scaler.transform(current_window[self.target_cols])
        current_seq = np.hstack([last_window_feats, last_window_targets])
        future_feats_scaled = self.feat_scaler.transform(future_exog[self.feat_cols])
        input_tensor = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0)
        for i in range(steps):
            with torch.no_grad():
                pred_scaled = self.model(input_tensor)
                preds.append(pred_scaled.numpy()[0])
                new_feat = torch.tensor(future_feats_scaled[i], dtype=torch.float32).unsqueeze(0)
                new_entry = torch.cat([new_feat, pred_scaled], dim=1).unsqueeze(1)
                input_tensor = torch.cat([input_tensor[:, 1:, :], new_entry], dim=1)
        return pd.DataFrame(
            self.target_scaler.inverse_transform(preds),
            columns=self.target_cols,
            index=future_exog.index,
        )


class CatBoostV3Wrapper(BaseForecaster):
    def __init__(self, path):
        self.model = ensure_skforecast_compat(joblib.load(path))
        self.target_cols = list(self.model.series_names_in_)
        self.exog_cols = list(self.model.exog_names_in_)
        self.depth_columns = self.target_cols
        self.max_lag = self.model.max_lag

    def predict(self, history_df, steps, future_exog):
        last_window = history_df[self.target_cols].tail(self.max_lag)
        predicts_long = self.model.predict(
            steps=steps,
            last_window=last_window,
            exog=future_exog[self.exog_cols],
        )
        df_res = predicts_long.reset_index().pivot(index="index", columns="level", values="pred")
        df_res.columns.name = None
        df_res.index = future_exog.index
        return df_res[self.target_cols]


class ARIMAWrapper(BaseForecaster):
    DEPTH_TO_FILE = {
        "48-1 (0)": "arima_cv_0.pkl",
        "48-1 (5)": "arima_cv_5.pkl",
        "48-1 (10)": "arima_cv_10.pkl",
    }

    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.depth_columns = list(self.DEPTH_TO_FILE.keys())

    def predict(self, history_df, steps, future_exog=None):
        results = {}
        for depth, filename in self.DEPTH_TO_FILE.items():
            try:
                pipe = joblib.load(Path(self.models_dir) / filename)
            except Exception as e:
                raise RuntimeError(
                    f"Не удалось загрузить ARIMA модель ({filename}). "
                    f"Возможна несовместимость pmdarima и numpy. "
                    f"Попробуйте: pip install 'numpy<2' --force-reinstall\n\nОшибка: {e}"
                ) from e
            results[depth] = pipe.predict(n_periods=steps).values
        return pd.DataFrame(results, index=future_exog.index)


class ExpSmoothingWrapper(BaseForecaster):
    depth_columns = ["48-Воздух (0)", "48-1 (3)", "48-1 (5)", "48-1 (10)"]

    @staticmethod
    def _fit_best(series: pd.Series):
        best_score = np.inf
        best_model = None
        seasonal_periods = 365
        train_shifted = series + (np.abs(series.min()) + 1)
        for trend in ["add", "mul", None]:
            for season in ["add", "mul", None]:
                if season is None and trend is None:
                    continue
                try:
                    fold_size = seasonal_periods
                    train_end_index = fold_size * 2 + 1
                    sum_mse = 0.0
                    iterations = 0
                    while train_end_index + fold_size < len(train_shifted):
                        model = ExponentialSmoothing(
                            train_shifted[:train_end_index],
                            seasonal_periods=seasonal_periods,
                            trend=trend,
                            seasonal=season,
                            initialization_method="heuristic",
                        ).fit()
                        pred = model.forecast(fold_size)
                        sum_mse += mean_squared_error(
                            train_shifted[train_end_index : train_end_index + fold_size],
                            pred,
                        )
                        iterations += 1
                        train_end_index += fold_size
                    if iterations == 0:
                        continue
                    score = sum_mse / iterations
                    if score < best_score:
                        best_score = score
                        best_model = ExponentialSmoothing(
                            train_shifted,
                            seasonal_periods=seasonal_periods,
                            trend=trend,
                            seasonal=season,
                            initialization_method="heuristic",
                        ).fit()
                except Exception:
                    continue
        if best_model is None:
            best_model = ExponentialSmoothing(
                train_shifted,
                seasonal_periods=seasonal_periods,
                trend="add",
                seasonal="add",
                initialization_method="heuristic",
            ).fit()
        shift = np.abs(series.min()) + 1
        return best_model, shift

    def predict(self, history_df, steps, future_exog):
        results = {}
        for col in self.depth_columns:
            if col not in history_df.columns:
                continue
            series = history_df[col].dropna()
            model, shift = self._fit_best(series)
            results[col] = model.forecast(steps) - shift
        return pd.DataFrame(results, index=future_exog.index)


class LinearRegressionWrapper(BaseForecaster):
    depth_columns = ["48-1 (1)", "48-1 (3)", "48-1 (5)", "48-1 (10)"]
    feature_cols = ["day_year_sin", "day_year_cos"]

    def predict(self, history_df, steps, future_exog):
        results = {}
        for col in self.depth_columns:
            if col not in history_df.columns:
                continue
            lr = LinearRegression()
            lr.fit(history_df[self.feature_cols], history_df[col])
            results[col] = lr.predict(future_exog[self.feature_cols])
        return pd.DataFrame(results, index=future_exog.index)


MODEL_CONFIG = {
    "XGBoost_v7": {
        "path": "xgboost/models/xgb_v7",
        "wrapper": XGBoostV7Wrapper,
        "preprocess": lambda df: process_data(df, include_is_anomaly=True),
        "min_history": 30,
    },
    "LSTM_v1": {
        "path": "rnn/models/lstm_v1.pkl",
        "wrapper": LSTMV1Wrapper,
        "preprocess": lambda df: process_data(df, include_is_anomaly=True),
        "min_history": 14,
    },
    "CatBoost_v3": {
        "path": "catboost/models/catboost_v3",
        "wrapper": CatBoostV3Wrapper,
        "preprocess": lambda df: process_data(df, include_is_anomaly=False),
        "min_history": 10,
    },
    "ARIMA": {
        "path": "furier_arima/models/diff_1",
        "wrapper": ARIMAWrapper,
        "preprocess": lambda df: df,
        "min_history": 365,
        "data_file": "data/processed/train, ZK 68, (48-1, 48-air), 27.11.20-31.12.24.csv",
    },
    "Exponential Smoothing": {
        "wrapper": ExpSmoothingWrapper,
        "preprocess": lambda df: df,
        "min_history": 730,
        "data_file": "data/processed/ZK 68, (48-1, 48-air), 27.11.20-15.12.25.csv",
    },
    "Linear Regression": {
        "wrapper": LinearRegressionWrapper,
        "preprocess": process_linear_regression,
        "min_history": 30,
        "data_file": "data/processed/ZK 68, (48-1, 48-air), 27.11.20-15.12.25.csv",
    },
}
