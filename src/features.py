import pandas as pd

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