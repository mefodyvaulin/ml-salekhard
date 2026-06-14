import sys
from pathlib import Path
import numpy as np
import pandas as pd
if str(Path(__file__).parent.parent.resolve()) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from src.features import cycle


def prepare_dataframe(data):
    df = data.copy()
    if "Дата" in df.columns:
        df["Дата"] = pd.to_datetime(df["Дата"])
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index("Дата", drop=False)
    else:
        df.index = pd.to_datetime(df.index)
        df["Дата"] = df.index
    df["Год"] = df.index.year
    df["Месяц"] = df.index.month
    df["День"] = df.index.day
    return df.asfreq("D")


def interpolate_numeric(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df
    interpolated = df[numeric_cols].interpolate(method="akima").ffill().bfill().round(2)
    df[numeric_cols] = interpolated
    return df


def process_data(data, include_is_anomaly = False):
    df = prepare_dataframe(data)
    df["day_of_year"] = df["Дата"].dt.dayofyear
    df = cycle(df, "Месяц", 12)
    df = cycle(df, "day_of_year", 365)
    if include_is_anomaly:
        df["is_anomaly"] = np.where(df["Дата"] <= "2021-05-17", 1, 0)
    df = df.drop(columns=["Дата"])
    return interpolate_numeric(df)


def process_linear_regression(data):
    df = prepare_dataframe(data)
    df["day_of_year"] = df["Дата"].dt.dayofyear
    df["day_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return interpolate_numeric(df.drop(columns=["Дата"]))


def build_future_frame(start_date, horizon):
    future_dates = pd.date_range(start=start_date, periods=horizon, freq="D")
    return pd.DataFrame({"Дата": future_dates}, index=future_dates)


def ensure_skforecast_compat(forecaster):
    if not hasattr(forecaster, "regressor") and hasattr(forecaster, "estimator"):
        forecaster.regressor = forecaster.estimator
    if not hasattr(forecaster, "regressors_") and hasattr(forecaster, "estimators_"):
        forecaster.regressors_ = forecaster.estimators_
    return forecaster
