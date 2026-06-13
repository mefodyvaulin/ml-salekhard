import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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
    
def show_acf_pacf(series: pd.Series, lags = 400):
    """Строит графики ACF и PACF для ряда"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series, lags=lags, ax=ax1)
    ax1.set_title(f'ACF: {series.name}')
    plot_pacf(series, lags=lags, ax=ax2)
    ax2.set_title(f'PACF: {series.name}')
    plt.show()


def plot_forecast(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    title="Прогноз vs Истина",
    metrics_dict: dict = None
):
    """Строит график сравнения истинных значений и прогноза"""
    plt.figure(figsize=(12, 5))
    
    plt.plot(y_true.index, y_true.values, label='Истинные', color='blue', linewidth=1.5)
    plt.plot(y_pred.index, y_pred.values, label='Прогноз', color='red', linestyle='--', linewidth=1.5)
    
    plt.xlabel('Дата')
    plt.ylabel('Температура')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    if metrics_dict:
        metrics_text = '\n'.join([f'{key}: {value:.4f}' for key, value in metrics_dict.items()])
        plt.gca().text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
    plt.tight_layout()
    plt.show()
    
def plot_forecast_with_train(
    X_train,
    y_true,
    y_pred,
    title="Прогноз vs Истина",
    metrics_dict: dict = None
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    if not y_pred.index.equals(y_true.index):
        y_pred.index = y_true.index
    
    ax1.plot(y_true.index, y_true.values, label='Истинные', color='blue', linewidth=1.5)
    ax1.plot(y_pred.index, y_pred.values, label='Прогноз', color='red', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Дата')
    ax1.set_ylabel('Температура')
    ax1.set_title('Тестовый период')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.plot(X_train.index, X_train.values, 
             label='Исторические (X_train)', color='green', linewidth=1.5, alpha=0.7)
    ax2.plot(y_true.index, y_true.values, 
             label='Истинные (тест)', color='blue', linewidth=1.5)
    ax2.plot(y_pred.index, y_pred.values, 
             label='Прогноз', color='red', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Дата')
    ax2.set_ylabel('Температура')
    ax2.set_title('Полный период (история + тест)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    if metrics_dict:
        metrics_text = '\n'.join([f'{key}: {value:.4f}' for key, value in metrics_dict.items()])
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()