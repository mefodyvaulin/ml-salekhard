import optuna
import xgboost as xgb
from skforecast.recursive import ForecasterRecursive
from sklearn.metrics import mean_squared_error
import numpy as np


def search_params(df_train, df_val, exog_cols, target_col):
    y_train = df_train[target_col]
    y_val = df_val[target_col]
    X_train = None
    X_val = None
    if exog_cols is not None:
        X_train = df_train[exog_cols]
        X_val = df_val[exog_cols]
    def objective(trial):
        num_lags = trial.suggest_int('lags', 1, 30)

        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.0005, 0.7, log=True),
            'max_depth': trial.suggest_int('max_depth', 1, 25),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-10, 100, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-10, 100, log=True),
            'gamma': trial.suggest_float('gamma', 1e-10, 10, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 20),
            'random_state': 42,
            'verbosity': 0,
            'n_jobs': -1
        }

        model = ForecasterRecursive(xgb.XGBRegressor(**param), 
                                    lags = num_lags)

        model.fit(
            y=y_train,
            exog=X_train
        )

        predictions = model.predict(
            steps=len(df_val),
            exog=X_val
        )

        rmse = np.sqrt(mean_squared_error(y_val, predictions))

        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=500)

    return study