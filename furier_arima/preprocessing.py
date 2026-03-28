import numpy as np
import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt


df = pd.read_csv("D:/monitoring_data.csv", sep=';', decimal=',', encoding='utf-8-sig')
df['45-2 (1)'] = df['45-2 (1)'].interpolate(method='spline', order=3)
df['44-Воздух (0)'] = df['44-Воздух (0)'].interpolate(method='spline', order=3)

df.to_csv("data.csv", index=False)
