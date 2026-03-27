import pandas as pd

def extract_year_month_day(df: pd.DataFrame) -> pd.DataFrame:
    """Преобразует столбец с датой в три столбца - год, месяц, день"""
    if (df['Дата'].dtype != 'datetime64[ns]'):
        df['Дата'] = pd.to_datetime(df['Дата'])

    df['Год'] = df['Дата'].dt.year
    df['Месяц'] = df['Дата'].dt.month
    df['День'] = df['Дата'].dt.day
    
    moved_cols_names = ['Год', 'Месяц', 'День']
    moved_cols = df[moved_cols_names].copy()
    df = df.drop(columns=moved_cols_names)

    for i, column in enumerate(moved_cols_names):
        df.insert(i, column, moved_cols[column])

    return df