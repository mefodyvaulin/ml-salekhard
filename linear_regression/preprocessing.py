import pandas as pd
import numpy as np


file_path = "../data/processed/ZK 68, (48-1, 48-air), 27.11.20-15.12.25.csv"
data = pd.read_csv(file_path)
data['Дата'] = pd.to_datetime(data['Дата'])

def preprocess_cycl_scale(df):
    df = df.copy()
    df['month_sin'] = np.sin(2 * np.pi * df['Месяц'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Месяц'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['День'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['День'] / 31)
    features = ['Год', 'month_sin', 'month_cos', 'day_sin', 'day_cos']
    return df, features

data_proc, base_feat = preprocess_cycl_scale(data)