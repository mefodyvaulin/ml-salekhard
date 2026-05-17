import pandas as pd


class MultiVariateForecaster:
    def __init__(self, forecasters):
        self.forecasters = {}
        for forecaster in forecasters:
            self.forecasters[forecaster.level] = forecaster

    def fit(self, df, exog = None):
        for forecaster in self.forecasters.values():
            forecaster.fit(df, exog = exog)
        return self

    def predict(self, steps, last_window, exog=None):
        predictions = []
        last_window = last_window.copy()

        for i in range(steps):
            step_predictions = {}
            exog_step = exog.iloc[[i]] if exog is not None else None
            for forecaster in self.forecasters.values():
                pred = forecaster.predict(steps=1, last_window=last_window, exog=exog_step)
                step_predictions[forecaster.level] = pred.iloc[0, 1]
            predictions.append(step_predictions)

            next_date = last_window.index[-1] + pd.Timedelta(days=1)
            new_row = pd.DataFrame([step_predictions], index=[next_date])
            new_row.index.freq = last_window.index.freq
            last_window = pd.concat([last_window, new_row])

        result = pd.DataFrame(predictions)
        
        return result