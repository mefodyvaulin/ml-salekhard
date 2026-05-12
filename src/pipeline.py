import pandas as pd
from datetime import datetime, timedelta
from src.visualization import plot_forecast
from src.preprocessing import hampel_filter_df_outliers
from skforecast.recursive import ForecasterRecursive
from sklearn.metrics import root_mean_squared_error
import joblib
from catboost import CatBoostRegressor


class ForecasterRecursivePipeline:
    def __init__(
        self,
        model_name: str,
        target_col: str,
        exog_cols: list[str],
        val_split_date: str,
        start_date: str,
        feature_transform_func = None,
        optimize_func = None,
        clean_outliers: bool = False
    ):
        self.model_name = model_name.lower()
        self.target_col = target_col
        self.exog_cols = exog_cols
        
        self.val_split_date = val_split_date
        self.start_date = start_date
        self.feature_transform_func = feature_transform_func
        self.optimize_func = optimize_func
        self.clean_outliers = clean_outliers
        
        self.best_params = None
        self.model = None
        self.rmse = None
        
    def _split_data(self, df_train_full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        dt = datetime.strptime(self.val_split_date, "%Y-%m-%d")
        prev_day = dt - timedelta(days=1)
        prev_day_str = prev_day.strftime("%Y-%m-%d")
        df_train = df_train_full[:prev_day_str].copy()
        df_val = df_train_full[self.val_split_date:].copy()
        return df_train, df_val
    
    def _preprocess_data(self, df: pd.DataFrame, is_train_data: bool) -> pd.DataFrame:
        df = df.copy()
        df = df.asfreq('D')
        
        if self.feature_transform_func is not None:
            df = self.feature_transform_func(df, self.start_date)

        df = df.interpolate(method='akima').ffill().bfill().round(2)

        if self.clean_outliers and is_train_data:
            df = hampel_filter_df_outliers(df)
            df = df.interpolate(method='akima').ffill().bfill().round(2)

        return df

    def _run_optimization(self, df_train: pd.DataFrame, df_val: pd.DataFrame):
        study = self.optimize_func(df_train, df_val, self.exog_cols, self.target_col)
        self.best_params = study.best_params
    
    def run_pipeline(self, df_train_full: pd.DataFrame, df_test: pd.DataFrame):
        df_train, df_val = self._split_data(df_train_full)
        
        df_train_full = self._preprocess_data(df_train_full, True)
        df_train = self._preprocess_data(df_train, True)
        df_val = self._preprocess_data(df_val, False)
        df_test = self._preprocess_data(df_test, False)
        
        self._run_optimization(df_train, df_val)
        
        self._fit_best_model(df_train_full)
        
        self.evaluate(df_test)
        
        return self        

    def _fit_best_model(self, df_train_full: pd.DataFrame):
        best_params_copy = self.best_params.copy()
        lags = best_params_copy.pop('lags')
        
        if self.model_name == 'catboost':
            best_params_copy['bootstrap_type'] = 'Bernoulli'            
            regressor = CatBoostRegressor(**best_params_copy, random_state=42, verbose=False)

        best_model = ForecasterRecursive(regressor, lags=lags)
        best_model.fit(y = df_train_full[self.target_col],
                       exog = df_train_full[self.exog_cols])
        self.model = best_model

    def evaluate(self, df_test: pd.DataFrame):
        y_pred = self.model.predict(steps=len(df_test),
                                    exog=df_test[self.exog_cols])
        self.rmse = root_mean_squared_error(df_test[self.target_col], y_pred)
        print(f'RMSE на тестовых данных: {self.rmse}')
        plot_forecast(df_test[self.target_col], y_pred)

    def save_model(self, path: str):
        if self.model is not None:
            joblib.dump(self.model, path)
            print(f'Модель сохранена: {path}')