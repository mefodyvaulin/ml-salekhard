import pandas as pd
import numpy as np


file_path = "../data/processed/ZK 68, (48-1, 48-air), 27.11.20-15.12.25.csv"
data = pd.read_csv(file_path)
data['Дата'] = pd.to_datetime(data['Дата'])

def preprocess_cycl_scale(df):
    df = df.copy()
    df['day_of_year'] = df['Дата'].dt.dayofyear
    df['day_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    feat = ['day_year_sin', 'day_year_cos']
    return df, feat

data_proc, base_feat = preprocess_cycl_scale(data)