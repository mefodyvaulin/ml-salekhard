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
