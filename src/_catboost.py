import optuna
from catboost import CatBoostRegressor
from skforecast.recursive import ForecasterRecursive
from sklearn.metrics import root_mean_squared_error
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def search_params(df_train, df_val, exog_cols, target_col, n_trials=10):
    y_train_full = df_train[target_col]
    y_val = df_val[target_col]
    X_train_full = None
    X_val = None
    if exog_cols is not None:
        X_train_full = df_train[exog_cols]
        X_val = df_val[exog_cols]
    def objective(trial):
        num_lags = trial.suggest_int('lags', 1, 10)

        param = {
            'iterations': trial.suggest_int('iterations', 100, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'depth': trial.suggest_int('depth', 1, 12),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 50, log=True),
            'bootstrap_type': 'Bernoulli',
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
            'random_strength': trial.suggest_float('random_strength', 1e-3, 10, log=True),
            'random_state': 42,
            'verbose': False,
            'thread_count': -1
        }

        model = ForecasterRecursive(CatBoostRegressor(**param), 
                                    lags = num_lags)

        model.fit(
            y=y_train_full,
            exog=X_train_full
        )

        predictions = model.predict(
            steps=len(df_val),
            exog=X_val
        )

        rmse = root_mean_squared_error(y_val, predictions)

        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    return study