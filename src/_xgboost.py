import optuna
import xgboost as xgb
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirectMultiVariate
from src.MultiVariateForecaster import MultiVariateForecaster
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler


def search_params(df_train, df_val, target_col, exog_cols = None, max_lags = 3, rollings = None, n_trials = 10):
    y_train = df_train[target_col]
    y_val = df_val[target_col]
    X_train = None
    X_val = None
    if exog_cols is not None:
        X_train = df_train[exog_cols]
        X_val = df_val[exog_cols]
    
    def objective(trial):
        num_lags = trial.suggest_int('lags', 1, max_lags)

        param = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.0005, 0.7, log=True),
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.4, 1.0),
            
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10, log=True),
            'gamma': trial.suggest_float('gamma', 1e-5, 5, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 20),
            'random_state': 42,
            'verbosity': 0,
            'n_jobs': -1
        }

        model = ForecasterRecursive(xgb.XGBRegressor(**param), 
                                    lags=num_lags,
                                    window_features=rollings)

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
    study.optimize(objective, n_trials=n_trials)

    return study

def search_params_mult(df_train, df_val, target_cols, exog_cols = None, max_lags = 5, max_neighbour_lags = 3, rollings = None, n_trials = 50):
    y_train = df_train[target_cols]
    y_val = df_val[target_cols]
    X_train = None
    X_val = None
    if exog_cols is not None:
        X_train = df_train[exog_cols]
        X_val = df_val[exog_cols]
        
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.0005, 0.7, log=True),
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.4, 1.0),
            
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10, log=True),
            'gamma': trial.suggest_float('gamma', 1e-5, 5, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 20),
            'random_state': 42,
            'verbosity': 0,
            'n_jobs': -1
        }
        
        models = []
        col = target_cols[0]
        lags_dict = {c: None for c in target_cols}
        lags_dict[col] = trial.suggest_int(f'lags_{col}', 1, max_lags)
        models.append(ForecasterDirectMultiVariate(estimator=xgb.XGBRegressor(**param),
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
            models.append(ForecasterDirectMultiVariate(estimator=xgb.XGBRegressor(**param),
                                                   level = col,
                                                   steps = 1,
                                                   lags = lags_dict,
                                                   window_features=rollings))
            
            
        forecaster = MultiVariateForecaster(models)
        forecaster.fit(y_train, exog=X_train)
        
        predictions = forecaster.predict(steps=len(df_val),last_window=y_train, exog=X_val)
        
        return rmse_mult(y_val, predictions)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    return study
        
        
def rmse_mult(y_true, y_pred):
    total_mse = 0
    for col in y_true.columns:
        scaler = StandardScaler()
        scaler.fit(y_true[col].values.reshape(-1, 1))
        
        y_true_scaled = scaler.transform(y_true[col].values.reshape(-1, 1))
        y_pred_scaled = scaler.transform(y_pred[col].values.reshape(-1, 1))
        
        mse = mean_squared_error(y_true_scaled, y_pred_scaled)
        total_mse += mse
    
    return np.sqrt(total_mse / len(y_true.columns))
