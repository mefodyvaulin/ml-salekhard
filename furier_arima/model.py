import math
import numpy as np
from pmdarima import pipeline
from pmdarima import preprocessing as ppc
from pmdarima import arima
from pmdarima.model_selection import RollingForecastCV, cross_validate
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


def grid_search(train, d, min_p, max_p, min_q, max_q, min_k, max_k, X=None):
    best_pipe = None
    best_aic = math.inf
    best_k = 0
    for k in range(min_k, max_k + 1):
        for p in range(min_p, max_p + 1):
            for q in range(min_q, max_q + 1):
                print(f'k = {k} p = {p} q = {q}')
                pipe = pipeline.Pipeline([
                    ("fourier", ppc.FourierFeaturizer(m=365, k=k)),
                    ("arima", arima.ARIMA((p, d, q), suppress_warnings=False, maxiter=1000))
                ])

                pipe.fit(train, X)
                model = pipe.named_steps['arima']
                aic = model.aic()
                print(f'AIC = {aic}')
                if aic < best_aic:
                    best_pipe = pipe
                    best_aic = aic
                    best_k = k

    print(f'Best Model = {best_pipe.named_steps['arima']}, k = {best_k}')

    return best_pipe


def grid_search_cv(train, d, min_p, max_p, min_q, max_q, min_k, max_k, h, step, initial, X=None):
    best_pipe = None
    best_mse = math.inf
    best_k = 0
    cv = RollingForecastCV(h, step, initial)
    for k in range(min_k, max_k + 1):
        for p in range(min_p, max_p + 1):
            for q in range(min_q, max_q + 1):
                print(f'k = {k} p = {p} q = {q}')
                pipe = pipeline.Pipeline([
                    ("fourier", ppc.FourierFeaturizer(m=365, k=k)),
                    ("arima", arima.ARIMA((p, d, q), suppress_warnings=False, maxiter=1000))
                ])
                res = cross_validate(pipe, train, X, scoring='mean_squared_error', cv=cv, verbose=2,
                                     error_score=math.inf)
                print(res['test_score'])
                mse = np.mean(res['test_score'])
                print(f'MSE = {mse})')
                if mse < best_mse:
                    best_mse = mse
                    best_pipe = pipe
                    best_k = k

    best_pipe.fit(train, X)
    print(f'Best Model = {best_pipe.named_steps['arima']}, k = {best_k}')
    return best_pipe


def save_pipeline(pipe, path):
    joblib.dump(pipe, path)


def get_pipe(path):
    return joblib.load(path)


def plot_forecast(train, test, forecast, conf_int):
    plt.figure(figsize=(14, 6))

    if hasattr(train, 'index'):
        plt.plot(train.index, train, label='Train', color='blue', alpha=0.6)
        plt.plot(test.index, test, label='Actual', color='green', linewidth=1.5)
        plt.plot(test.index, forecast, label='Forecast', color='red', linewidth=2)
        plt.fill_between(test.index, conf_int[:, 0], conf_int[:, 1],
                         color='red', alpha=0.2, label='95% CI')
        plt.xlabel('Дата')
        plt.xticks(rotation=45)
    else:
        plt.plot(train, label='Train', color='blue', alpha=0.6)
        plt.plot(test, label='Actual', color='green', linewidth=1.5)
        plt.plot(forecast, label='Forecast', color='red', linewidth=2)
        plt.fill_between(range(len(test)), conf_int[:, 0], conf_int[:, 1],
                         color='red', alpha=0.2)
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
