import pandas as pd
from model import*
import pmdarima as pm
import matplotlib.pyplot as plt


df = pd.read_csv('data.csv')
train = df['44-Воздух (0)'][:-365]
test = df['44-Воздух (0)'][-365:]

_, trend, season, noise = pm.arima.decompose(train, 'additive', 365)
pm.utils.decomposed_plot((_, trend, season, noise), figure_kwargs={})

train_seasonless = train - season
train_seasonless_diff = pm.utils.diff(train_seasonless)
a = pm.utils.diff(train)
b=pm.utils.diff(a, 365)
plt.plot(a)
pm.utils.plot_acf(b)
pm.utils.plot_pacf(b)



#pipe = grid_search_cv(train, 1, 0,3,0,3, 10, 10,365,182,365)
pipe = grid_search(train, 1, 0,3,0,3, 5, 10)
forecast, conf_int = pipe.predict(n_periods=len(test), return_conf_int=True)

plot_forecast(train, test, forecast, conf_int)
print_metrics(forecast, test)
