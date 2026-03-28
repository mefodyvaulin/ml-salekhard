import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def show_series_stl(series: pd.Series):
    """Строит графики STL-декомпозиции заданного ряда"""
    from statsmodels.tsa.seasonal import STL
    
    stl = STL(series, period=365, robust=True)
    result = stl.fit()

    trend = result.trend
    seasonal = result.seasonal
    resid = result.resid
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 15), sharex=True)

    series.plot(ax=ax1, color='royalblue', label='Исходные данные')
    ax1.set_ylabel('Исходные данные', size=16)
    ax1.legend(loc='upper left')
    ax1.set_title(series.name, size=18)
    
    trend.plot(ax=ax2, color='red', label='Тренд')
    ax2.set_ylabel('Тренд', size=16)
    ax2.legend(loc='upper left')
    
    seasonal.plot(ax=ax3, color='green', label='Сезонность')
    ax3.set_ylabel('Сезонность', size=16)
    ax3.legend(loc='upper left')
    
    resid.plot(ax=ax4, color='purple', label='Ошибка')
    ax4.set_ylabel('Ошибка', size=16)
    ax4.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()