import optuna
from catboost import CatBoostRegressor
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
from sklearn.metrics import root_mean_squared_error
from src.MultiVariateForecaster import MultiVariateForecaster
import numpy as np
from skforecast.direct import ForecasterDirectMultiVariate
from sklearn.preprocessing import StandardScaler


def search_params_forecaster_recursive(df_train, df_val, exog_cols, target_col, n_trials=10):
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

        y_pred = model.predict(
            steps=len(df_val),
            exog=X_val
        )

        rmse = root_mean_squared_error(y_val, y_pred)

        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    return study

def search_params_mult(
    df_train,
    df_val,
    target_cols,
    exog_cols=None,
    max_lags=5,
    max_neighbour_lags=3,
    rollings=None,
    n_trials=50
):
    y_train_dict = df_train[target_cols]
    y_val = df_val[target_cols]
    X_train = None
    X_val = None
    if exog_cols is not None:
        X_train = df_train[exog_cols]
        X_val = df_val[exog_cols]
        
    def objective(trial):
        param = {
            'iterations': trial.suggest_int('iterations', 100, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'depth': trial.suggest_int('depth', 1, 10),
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
        
        models = []
        col = target_cols[0]
        lags_dict = {c: None for c in target_cols}
        lags_dict[col] = trial.suggest_int(f'lags_{col}', 1, max_lags)
        models.append(ForecasterDirectMultiVariate(estimator=CatBoostRegressor(**param),
                                                   level = col,
                                                   steps = 1,
                                                   lags = lags_dict,
                                                   window_features=rollings))
        
        for i in range(1, len(target_cols)):
            col = target_cols[i]
            neighbor_col = target_cols[i - 1]
            lags_dict = {c: None for c in target_cols}
            lags_dict[col] = trial.suggest_int(f'lags_{col}', 1, max_lags)
            lags_dict[neighbor_col] = trial.suggest_int(f'neighbor_lags_{col}', 1, max_neighbour_lags)
            models.append(ForecasterDirectMultiVariate(estimator=CatBoostRegressor(**param),
                                                   level = col,
                                                   steps = 1,
                                                   lags = lags_dict,
                                                   window_features=rollings))
            
            
        forecaster = MultiVariateForecaster(models)
        forecaster.fit(y_train_dict, exog=X_train)
        
        y_pred = forecaster.predict(steps=len(df_val),last_window=y_train_dict, exog=X_val)
        
        return rmse_mult(y_val, y_pred)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    return study

def search_params_recursive_multi_series(
    df_train,
    df_val,
    target_cols,
    exog_cols,
    max_lags=10,
    n_trials=10
):
    y_train_dict = {col: df_train[col] for col in target_cols}
    y_val = df_val[target_cols]
    
    X_train = df_train[exog_cols]
    X_val = df_val[exog_cols]
    
    def objective(trial):
        param = {
            'iterations': trial.suggest_int('iterations', 100, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'depth': trial.suggest_int('depth', 1, 10),
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
        
        lags = trial.suggest_int('lags', 1, max_lags)
        
        forecaster = ForecasterRecursiveMultiSeries(
            estimator=CatBoostRegressor(**param),
            lags=lags,
        )
        
        forecaster.fit(series=y_train_dict, exog=X_train)
        
        y_pred_long = forecaster.predict(steps=len(df_val), exog=X_val)
        y_pred = y_pred_long.reset_index().pivot(index='index', columns='level', values='pred')
        y_pred.columns.name = None
        return rmse_mult(y_val, y_pred)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    return study

def rmse_mult(y_true, y_pred):
    total_rmse = 0
    for col in y_true.columns:
        scaler = StandardScaler()
        scaler.fit(y_true[col].values.reshape(-1, 1))
        
        y_true_scaled = scaler.transform(y_true[col].values.reshape(-1, 1))
        y_pred_scaled = scaler.transform(y_pred[col].values.reshape(-1, 1))
        
        rmse = root_mean_squared_error(y_true_scaled, y_pred_scaled)
        total_rmse += rmse
    
    return total_rmse / len(y_true.columns)