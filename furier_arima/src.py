import math
import numpy as np
from pmdarima import pipeline
from pmdarima import preprocessing as ppc
from pmdarima import arima
from pmdarima.model_selection import RollingForecastCV, cross_validate
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


def grid_search(train, d, p_range, q_range, k_range, X=None):
    best_pipe = None
    best_aic = math.inf
    best_k = 0
    for k in k_range:
        for p in p_range:
            for q in q_range:
                pipe = pipeline.Pipeline([
                    ("fourier", ppc.FourierFeaturizer(m=365, k=k)),
                    ("arima", arima.ARIMA((p, d, q), suppress_warnings=False, maxiter=1000))
                ])

                pipe.fit(train, X)
                model = pipe.named_steps['arima']
                aic = model.aic()
                if aic < best_aic:
                    best_pipe = pipe
                    best_aic = aic
                    best_k = k

    print(f"Best Model = {best_pipe.named_steps['arima']}, k = {best_k}")

    return best_pipe


def grid_search_cv(train, d, p_range, q_range, k_range, cv, X=None):
    best_pipe = None
    best_mse = math.inf
    best_k = 0
    for k in k_range:
        for p in p_range:
            for q in q_range:
                pipe = pipeline.Pipeline([
                    ("fourier", ppc.FourierFeaturizer(m=365, k=k)),
                    ("arima", arima.ARIMA((p, d, q), suppress_warnings=False, maxiter=1000))
                ])
                res = cross_validate(pipe, train, X, scoring='mean_squared_error', cv=cv, verbose=0,
                                     error_score=math.inf)
                mse = np.mean(res['test_score'])
                if mse < best_mse:
                    best_mse = mse
                    best_pipe = pipe
                    best_k = k

    best_pipe.fit(train, X)
    print(f"Best Model = {best_pipe.named_steps['arima']}, k = {best_k}")
    return best_pipe

def plot_forecast(train, test, forecast):
    plt.figure(figsize=(14, 6))

    if hasattr(train, 'index'):
        plt.plot(train.index, train, label='Train', color='blue', alpha=0.6)
        plt.plot(test.index, test, label='Actual', color='green', linewidth=1.5)
        plt.plot(test.index, forecast, label='Forecast', color='red', linewidth=2)
        plt.xlabel('Дата')
        plt.xticks(rotation=45)
    else:
        plt.plot(train, label='Train', color='blue', alpha=0.6)
        plt.plot(test, label='Actual', color='green', linewidth=1.5)
        plt.plot(forecast, label='Forecast', color='red', linewidth=2)
        plt.xlabel('Наблюдение')

    plt.title('Прогноз: ARIMA + FourierFeaturizer')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_metrics(forecast, test):
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    print(f"\nMAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
