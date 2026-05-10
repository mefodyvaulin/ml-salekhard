import pandas as pd
import numpy as np

def extract_year_month_day(df: pd.DataFrame) -> pd.DataFrame:
    """Преобразует столбец с датой в три столбца - год, месяц, день.
    
    Args:
        df (pd.DataFrame): Датафрейм с индексом datetime
    Returns:
        df (pd.DataFrame): Датафрейм с выделенным годом, месяцем и днём
    """
    df = df.copy()
    if (df.index.dtype not in ['<M8[ns]' or 'datetime64[ns]']):
        raise ValueError('Индекс датафрейма должен быть в формате datetime')
    
    df['Дата'] = pd.to_datetime(df.index)
    
    df['Год'] = df['Дата'].dt.year
    df['Месяц'] = df['Дата'].dt.month
    df['День'] = df['Дата'].dt.day
    
    df = df.drop(columns=['Дата'])
    moved_cols_names = ['Год', 'Месяц', 'День']
    moved_cols = df[moved_cols_names].copy()
    df = df.drop(columns=moved_cols_names)

    for i, column in enumerate(moved_cols_names):
        df.insert(i, column, moved_cols[column])

    return df

def cycle_day_month(df: pd.DataFrame) -> pd.DataFrame:
    """Зацикливает день и месяц, добавляя новые признаки"""
    df_copy = df.copy()
    
    month = df_copy['Месяц']
    df_copy['Месяц_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
    df_copy['Месяц_cos'] = np.cos(2 * np.pi * (month - 1) / 12)
    
    days_map = {
        1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
        7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
    }
    days_in_month = month.map(days_map)
    
    day_normalized = (df_copy['День'] - 1) / days_in_month
    df_copy['День_sin'] = np.sin(2 * np.pi * day_normalized)
    df_copy['День_cos'] = np.cos(2 * np.pi * day_normalized)

    
    return df_copy

def cycle(df: pd.DataFrame, col_name: str, period: int) -> pd.DataFrame:
    """Зацикливает столбец с указанным периодом, добавляя новые признаки"""
    df = df.copy()
    
    col = df[col_name]
    df[f'{col_name}_sin'] = np.sin(2 * np.pi * (col - 1) / period)
    df[f'{col_name}_cos'] = np.cos(2 * np.pi * (col - 1) / period)

    return df

def stl_decompose_df(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Делает STL-декомпозицию заданных столбцов датафрейма"""
    from statsmodels.tsa.seasonal import STL
    
    df_res = df.copy()
    
    if (isinstance(columns, str)):
        columns = [columns]

    for col in columns:
        stl = STL(df[col], period=365, robust=True)
        result = stl.fit()

        trend = result.trend
        seasonal = result.seasonal
        resid = result.resid
        
        df_res = df_res.drop(columns=col)

        df_res[f'{col} trend'] = trend
        df_res[f'{col} seasonal'] = seasonal
        df_res[f'{col} resid'] = resid

    return df_res