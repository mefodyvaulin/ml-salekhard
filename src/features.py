import pandas as pd
import numpy as np
import re


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

def transform_depth(df: pd.DataFrame, air_name: str) -> pd.DataFrame:
    df = df.copy()
    
    df.columns = [col.replace(',', '.') for col in df.columns]
    
    depth_cols = [col for col in df.columns if re.search(r'\d+-\d+ \(\d+\.?\d*\)', col)]
    other_cols = [col for col in df.columns if col not in depth_cols and col != air_name]
    
    df_long = df.melt(
        id_vars=other_cols + [air_name], 
        value_vars=depth_cols,
        var_name='Исходная_глубина', 
        value_name='Температура'
    )

    df_long['Глубина'] = df_long['Исходная_глубина'].str.extract(r'\((\d+\.?\d*)\)').astype(float)

    air_df = df[other_cols + [air_name]].copy()
    air_df['Температура'] = air_df[air_name]
    air_df['Глубина'] = -1.0
    air_df = air_df.drop(columns=[air_name])

    df_final = pd.concat([air_df, df_long.drop(columns=['Исходная_глубина', air_name])], ignore_index=True)
    
    return df_final

def untransform_depth(df: pd.DataFrame, air_name: str) -> pd.DataFrame:
    """Разворачивает данные из длинного формата (long) обратно в широкий (wide).
    
    Args:
        df (pd.DataFrame): Датафрейм после transform_depth
        air_name (str): Название столбца для температуры воздуха (глубина -1.0)
    """
    df = df.copy()

    # Разделяем на температуру воздуха и остальные данные
    air_df = df[df['Глубина'] == -1.0].copy()
    depth_df = df[df['Глубина'] != -1.0].copy()

    # Возвращаем воздух: переименовываем Температуру обратно в air_name
    air_df[air_name] = air_df['Температура']
    air_df = air_df.drop(columns=['Глубина', 'Температура'])

    # Превращаем Глубину обратно в названия исходных столбцов
    # Нам нужно восстановить строку вида "номер-номер (число)"
    # Если точные исходные префиксы (например, "48-1") не сохранены, 
    # придется использовать шаблон или хранить отображение.
    # Предположим, мы восстанавливаем формат "Depth (число)" или просто "Глубина (число)"
    depth_df['col_name'] = depth_df['Глубина'].apply(lambda x: f'Глубина ({x})')

    # Разворачиваем (Pivot)
    other_cols = [col for col in depth_df.columns if col not in ['Глубина', 'Температура', 'col_name']]
    df_wide = depth_df.pivot(index=other_cols, columns='col_name', values='Температура').reset_index()
    df_wide.columns.name = None

    # Объединяем с данными о воздухе
    # Используем merge по общим колонкам (времени/признакам)
    final_df = pd.merge(air_df, df_wide, on=other_cols)

    return final_df