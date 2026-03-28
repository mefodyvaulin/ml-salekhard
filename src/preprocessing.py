import pandas as pd
import numpy as np

def create_gaps_in_df(df: pd.DataFrame, columns='all', num_blocks=6, block_size=3, seed=42) -> pd.DataFrame:
    """Создает пропуски в датафрейме в виде блоков"""
    df_with_gaps = df.copy()
    max_start_idx = len(df_with_gaps) - block_size

    np.random.seed(seed)
    start_indices = np.random.choice(range(max_start_idx), size=num_blocks, replace=False)

    if columns == 'all':
        col_indices = slice(None)
    elif isinstance(columns, str):
        col_indices = df_with_gaps.columns.get_loc(columns)
    elif isinstance(columns, list):
        col_indices = [df_with_gaps.columns.get_loc(col) for col in columns]
    else:
        raise ValueError("Неправильное значение для columns")

    for start in start_indices:
        df_with_gaps.iloc[start : start + block_size, col_indices] = np.nan

    return df_with_gaps


def calculate_mae_for_gaps(df_true: pd.DataFrame, df_pred: pd.DataFrame, mask: pd.DataFrame) -> float:
    """Считает MAE по всем пропускам во всех столбцах"""
    from sklearn.metrics import mean_absolute_error

    true_values = df_true.values[mask.values]
    pred_values = df_pred.values[mask.values]
    
    mae = mean_absolute_error(true_values, pred_values)
    return mae

def hampel_filter_df_outliers(df: pd.DataFrame, window_length=7, n_sigma=3) -> pd.DataFrame:
    """Очищает датасет от выбросов с помощью фильтра Хампеля"""
    from sktime.transformations.series.outlier_detection import HampelFilter
    
    filter = HampelFilter(window_length=window_length, n_sigma=n_sigma)
    df_clean = df.copy()
    for col in df_clean.columns:
        df_clean[col] = filter.fit_transform(df_clean[col])

    return df_clean