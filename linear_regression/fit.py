import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from linear_regression.preprocessing import data_proc, base_feat

temp_podp = '48-Воздух (0)'
depth = ['48-1 (1)', '48-1 (3)', '48-1 (5)', '48-1 (10)']
results = []
for target in depth:
    print(f"\n Предсказание для глубины: {target}")
    train_data = data_proc[data_proc['Год'] < 2025]
    test_data = data_proc[data_proc['Год'] == 2025]

    feat_sets = {
        "Без температуры подполья": base_feat,
        "С температурой подполья": base_feat + [temp_podp]
    }
    all_preds = {}

    for name, feat_lst in feat_sets.items():
        X_train = train_data[feat_lst]
        y_train = train_data[target]
        X_test = test_data[feat_lst]
        y_test = test_data[target]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        all_preds[name] = preds

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        print(f"{name}: MAE = {mae:.4f}, RMSE = {rmse:.4f}")
        results.append({
            'Depth': target,
            'Model': name,
            'MAE': mae,
            'RMSE': rmse
        })

    plt.figure(figsize=(14, 7))
    plt.plot(train_data['Дата'], train_data[target], label='Обучающие данные', color='blue', alpha=0.4)
    plt.plot(test_data['Дата'], test_data[target], label='Реальные данные', color='black', linewidth=2)
    colors = ['orange', 'green']
    for (name, preds), color in zip(all_preds.items(), colors):
        plt.plot(test_data['Дата'], preds, label=f'Прогноз: {name}', linestyle='--', linewidth=2, color=color)
    plt.title(f'Линейная регрессия: Прогноз температуры на глубине {target}')
    plt.xlabel('Дата')
    plt.ylabel('Температура')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

results_df = pd.DataFrame(results)
print("\nИтоговое сравнение:")
print(results_df)