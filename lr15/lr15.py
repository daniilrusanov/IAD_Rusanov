import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Статистичні моделі
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Машинне навчання
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

# ============================================================================
# 1. ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА ДАНИХ
# ============================================================================
print("\n1. ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА ДАНИХ")
print("-" * 80)

# Створення синтетичних даних (імітація продажів)
np.random.seed(42)
n_points = 365 * 2  # 2 роки даних

# Генерація дат
date_range = pd.date_range(start='2022-01-01', periods=n_points, freq='D')

# Компоненти часового ряду
trend = np.linspace(100, 200, n_points)  # Зростаючий тренд
seasonal = 30 * np.sin(2 * np.pi * np.arange(n_points) / 365)  # Річна сезонність
weekly = 10 * np.sin(2 * np.pi * np.arange(n_points) / 7)  # Тижнева сезонність
noise = np.random.normal(0, 5, n_points)  # Шум

# Об'єднання компонентів
sales = trend + seasonal + weekly + noise

# Створення DataFrame
df = pd.DataFrame({
    'date': date_range,
    'sales': sales
})
df.set_index('date', inplace=True)
# Явно встановлюємо частоту (Daily)
df.index.freq = 'D'

print(f"✓ Створено часовий ряд")
print(f"  Період: {df.index[0].date()} - {df.index[-1].date()}")
print(f"  Кількість спостережень: {len(df)}")
print(f"  Середнє значення: {df['sales'].mean():.2f}")
print(f"  Стандартне відхилення: {df['sales'].std():.2f}")
print(f"  Мінімум: {df['sales'].min():.2f}")
print(f"  Максимум: {df['sales'].max():.2f}")

# Перевірка пропущених значень
print(f"\n  Пропущені значення: {df.isnull().sum().sum()}")

# Статистика по місяцях
monthly_stats = df.resample('M')['sales'].agg(['mean', 'std', 'min', 'max'])
print("\n  Статистика по місяцях (перші 5):")
print(monthly_stats.head())

# ============================================================================
# 2. ВІЗУАЛІЗАЦІЯ ДАНИХ
# ============================================================================
print("\n2. ВІЗУАЛІЗАЦІЯ ЧАСОВОГО РЯДУ")
print("-" * 80)

fig = plt.figure(figsize=(18, 14))

# 2.1. Основний часовий ряд
ax1 = plt.subplot(4, 3, 1)
plt.plot(df.index, df['sales'], linewidth=1, alpha=0.8, color='#2E86AB')
plt.title('Часовий ряд продажів', fontsize=12, fontweight='bold')
plt.xlabel('Дата')
plt.ylabel('Продажі')
plt.grid(True, alpha=0.3)

# Додаємо ковзне середнє
rolling_mean = df['sales'].rolling(window=30).mean()
plt.plot(df.index, rolling_mean, linewidth=2, color='red',
         label='MA(30)', alpha=0.7)
plt.legend()

# 2.2. Гістограма розподілу
ax2 = plt.subplot(4, 3, 2)
plt.hist(df['sales'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(df['sales'].mean(), color='red', linestyle='--',
            linewidth=2, label=f'Середнє: {df["sales"].mean():.1f}')
plt.title('Розподіл значень продажів', fontsize=12, fontweight='bold')
plt.xlabel('Продажі')
plt.ylabel('Частота')
plt.legend()
plt.grid(True, alpha=0.3)

# 2.3. Box plot по місяцях
ax3 = plt.subplot(4, 3, 3)
df_monthly = df.copy()
df_monthly['month'] = df_monthly.index.month
monthly_data = [df_monthly[df_monthly['month'] == m]['sales'].values
                for m in range(1, 13)]
bp = plt.boxplot(monthly_data, labels=range(1, 13), patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
plt.title('Box plot по місяцях', fontsize=12, fontweight='bold')
plt.xlabel('Місяць')
plt.ylabel('Продажі')
plt.grid(True, alpha=0.3)

# ============================================================================
# 3. ДЕКОМПОЗИЦІЯ ЧАСОВОГО РЯДУ
# ============================================================================
print("\n3. ДЕКОМПОЗИЦІЯ ЧАСОВОГО РЯДУ")
print("-" * 80)

# Виконання декомпозиції
decomposition = seasonal_decompose(df['sales'], model='additive', period=365)

trend_component = decomposition.trend
seasonal_component = decomposition.seasonal
residual_component = decomposition.resid

print("✓ Декомпозиція виконана")
print(f"  Тренд: розмір {len(trend_component.dropna())}")
print(f"  Сезонність: період {365} днів")
print(f"  Залишки: std = {residual_component.std():.2f}")

# 3.1. Тренд
ax4 = plt.subplot(4, 3, 4)
plt.plot(df.index, trend_component, linewidth=2, color='#A23B72')
plt.title('Компонента тренду', fontsize=12, fontweight='bold')
plt.xlabel('Дата')
plt.ylabel('Тренд')
plt.grid(True, alpha=0.3)

# 3.2. Сезонність
ax5 = plt.subplot(4, 3, 5)
plt.plot(df.index[:365], seasonal_component[:365], linewidth=2, color='#F18F01')
plt.title('Сезонна компонента (перший рік)', fontsize=12, fontweight='bold')
plt.xlabel('Дата')
plt.ylabel('Сезонність')
plt.grid(True, alpha=0.3)

# 3.3. Залишки
ax6 = plt.subplot(4, 3, 6)
plt.plot(df.index, residual_component, linewidth=1, alpha=0.7, color='#6A994E')
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
plt.title('Залишкова компонента', fontsize=12, fontweight='bold')
plt.xlabel('Дата')
plt.ylabel('Залишки')
plt.grid(True, alpha=0.3)

# ============================================================================
# 4. ТЕСТ НА СТАЦІОНАРНІСТЬ
# ============================================================================
print("\n4. ТЕСТ НА СТАЦІОНАРНІСТЬ (AUGMENTED DICKEY-FULLER)")
print("-" * 80)

# ADF тест
adf_result = adfuller(df['sales'].dropna())
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")
print(f"Критичні значення:")
for key, value in adf_result[4].items():
    print(f"  {key}: {value:.3f}")

if adf_result[1] < 0.05:
    print("✓ Ряд є стаціонарним (p < 0.05)")
else:
    print("✗ Ряд НЕ є стаціонарним (p >= 0.05)")
    print("  Потрібне диференціювання")

# ============================================================================
# 5. ACF ТА PACF
# ============================================================================
print("\n5. АНАЛІЗ АВТОКОРЕЛЯЦІЇ")
print("-" * 80)

# 5.1. ACF
ax7 = plt.subplot(4, 3, 7)
plot_acf(df['sales'].dropna(), lags=40, ax=ax7)
plt.title('Функція автокореляції (ACF)', fontsize=12, fontweight='bold')

# 5.2. PACF
ax8 = plt.subplot(4, 3, 8)
plot_pacf(df['sales'].dropna(), lags=40, ax=ax8)
plt.title('Часткова автокореляція (PACF)', fontsize=12, fontweight='bold')

# ============================================================================
# 6. РОЗДІЛЕННЯ НА TRAIN/TEST
# ============================================================================
print("\n6. РОЗДІЛЕННЯ ДАНИХ НА TRAIN/TEST")
print("-" * 80)

# Розділення 80/20
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]

print(f"✓ Дані розділено:")
print(f"  Train: {len(train_data)} спостережень ({train_data.index[0].date()} - {train_data.index[-1].date()})")
print(f"  Test: {len(test_data)} спостережень ({test_data.index[0].date()} - {test_data.index[-1].date()})")

# Візуалізація розділення
ax9 = plt.subplot(4, 3, 9)
plt.plot(train_data.index, train_data['sales'], label='Train', color='blue', alpha=0.7)
plt.plot(test_data.index, test_data['sales'], label='Test', color='orange', alpha=0.7)
plt.title('Розділення на Train/Test', fontsize=12, fontweight='bold')
plt.xlabel('Дата')
plt.ylabel('Продажі')
plt.legend()
plt.grid(True, alpha=0.3)

# ============================================================================
# 7. МОДЕЛЬ ARIMA
# ============================================================================
print("\n7. ПОБУДОВА МОДЕЛІ ARIMA")
print("-" * 80)

# Підбір параметрів ARIMA (p, d, q)
# Використовуємо простіші параметри для кращої збіжності
print("Навчання моделі ARIMA(2,1,2)...")
try:
    arima_model = ARIMA(train_data['sales'], order=(2, 1, 2))
    arima_fitted = arima_model.fit(method_kwargs={'maxiter': 500, 'disp': False})

    print("✓ Модель ARIMA навчена")
    print(f"\nПараметри моделі:")
    print(f"  p (AR): 2")
    print(f"  d (I): 1")
    print(f"  q (MA): 2")
    print(f"  AIC: {arima_fitted.aic:.2f}")
    print(f"  BIC: {arima_fitted.bic:.2f}")
except Exception as e:
    print(f"⚠️ Помилка при навчанні ARIMA: {e}")
    print("Спроба з простішою моделлю ARIMA(1,1,1)...")
    arima_model = ARIMA(train_data['sales'], order=(1, 1, 1))
    arima_fitted = arima_model.fit(method_kwargs={'maxiter': 500, 'disp': False})

    print("✓ Модель ARIMA навчена")
    print(f"\nПараметри моделі:")
    print(f"  p (AR): 1")
    print(f"  d (I): 1")
    print(f"  q (MA): 1")
    print(f"  AIC: {arima_fitted.aic:.2f}")
    print(f"  BIC: {arima_fitted.bic:.2f}")

# Прогноз на тестову вибірку
arima_forecast = arima_fitted.forecast(steps=len(test_data))
arima_predictions = pd.Series(arima_forecast, index=test_data.index)

# Метрики для ARIMA
arima_mae = mean_absolute_error(test_data['sales'], arima_predictions)
arima_rmse = np.sqrt(mean_squared_error(test_data['sales'], arima_predictions))
arima_mape = np.mean(np.abs((test_data['sales'] - arima_predictions) / test_data['sales'])) * 100

print(f"\nМетрики ARIMA:")
print(f"  MAE: {arima_mae:.2f}")
print(f"  RMSE: {arima_rmse:.2f}")
print(f"  MAPE: {arima_mape:.2f}%")

# Збереження інформації про порядок моделі
arima_order = arima_fitted.model.order
print(f"\n  Використаний порядок моделі: ARIMA{arima_order}")

# ============================================================================
# 8. МОДЕЛЬ EXPONENTIAL SMOOTHING (ETS)
# ============================================================================
print("\n8. ПОБУДОВА МОДЕЛІ EXPONENTIAL SMOOTHING")
print("-" * 80)

print("Навчання моделі ETS...")
# Використовуємо тижневу сезонність (7 днів) замість річної
# або модель без сезонності, якщо даних недостатньо
try:
    ets_model = ExponentialSmoothing(
        train_data['sales'],
        seasonal_periods=7,  # Тижнева сезонність
        trend='add',
        seasonal='add',
        damped_trend=True
    )
    ets_fitted = ets_model.fit()
    print("✓ Модель ETS навчена (з тижневою сезонністю)")
except:
    # Якщо і це не працює, використовуємо простішу модель
    ets_model = ExponentialSmoothing(
        train_data['sales'],
        trend='add',
        seasonal=None,  # Без сезонності
        damped_trend=True
    )
    ets_fitted = ets_model.fit()
    print("✓ Модель ETS навчена (без сезонності)")

# Прогноз
ets_forecast = ets_fitted.forecast(steps=len(test_data))
ets_predictions = pd.Series(ets_forecast, index=test_data.index)

# Метрики для ETS
ets_mae = mean_absolute_error(test_data['sales'], ets_predictions)
ets_rmse = np.sqrt(mean_squared_error(test_data['sales'], ets_predictions))
ets_mape = np.mean(np.abs((test_data['sales'] - ets_predictions) / test_data['sales'])) * 100

print(f"\nМетрики ETS:")
print(f"  MAE: {ets_mae:.2f}")
print(f"  RMSE: {ets_rmse:.2f}")
print(f"  MAPE: {ets_mape:.2f}%")

# ============================================================================
# 9. МОДЕЛЬ ЛІНІЙНОЇ РЕГРЕСІЇ З ЛАГАМИ
# ============================================================================
print("\n9. МОДЕЛЬ МАШИННОГО НАВЧАННЯ (LINEAR REGRESSION)")
print("-" * 80)

# Створення лагових ознак з правильною логікою для прогнозування
def create_lagged_features(data, lags=7):
    """Створення лагових ознак"""
    df_lag = pd.DataFrame(index=data.index)
    df_lag['sales'] = data['sales']

    for i in range(1, lags + 1):
        df_lag[f'lag_{i}'] = data['sales'].shift(i)

    # Додаткові ознаки
    df_lag['rolling_mean_7'] = data['sales'].rolling(window=7).mean()
    df_lag['rolling_std_7'] = data['sales'].rolling(window=7).std()
    df_lag['day_of_week'] = data.index.dayofweek
    df_lag['day_of_year'] = data.index.dayofyear

    return df_lag.dropna()

# Навчання на тренувальних даних
train_features = create_lagged_features(train_data, lags=7)
X_train = train_features.drop('sales', axis=1)
y_train = train_features['sales']

print("✓ Створено лагові ознаки")
print(f"  Кількість ознак: {X_train.shape[1]}")

# Навчання моделі
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print("✓ Модель Linear Regression навчена")

# Прогноз: використовуємо останні дані з train для створення перших лагів
# Об'єднуємо останні 7 значень з train та test для коректних лагів
combined_data = pd.concat([train_data.tail(7), test_data])
test_features_combined = create_lagged_features(combined_data, lags=7)

# Беремо тільки ті рядки, що відповідають test_data
test_features = test_features_combined.loc[test_data.index]
test_features = test_features.dropna()

X_test = test_features.drop('sales', axis=1)
y_test = test_features['sales']

# Прогноз
lr_predictions = lr_model.predict(X_test)
lr_pred_series = pd.Series(lr_predictions, index=y_test.index)

# Метрики
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
lr_mape = np.mean(np.abs((y_test - lr_predictions) / y_test)) * 100

print(f"\nМетрики Linear Regression:")
print(f"  MAE: {lr_mae:.2f}")
print(f"  RMSE: {lr_rmse:.2f}")
print(f"  MAPE: {lr_mape:.2f}%")

# ============================================================================
# 10. ВІЗУАЛІЗАЦІЯ ПРОГНОЗІВ
# ============================================================================
print("\n10. ВІЗУАЛІЗАЦІЯ ПРОГНОЗІВ")
print("-" * 80)

# 10.1. Прогноз ARIMA
ax10 = plt.subplot(4, 3, 10)
plt.plot(train_data.index[-90:], train_data['sales'][-90:],
         label='Train', color='blue', linewidth=1.5)
plt.plot(test_data.index[:90], test_data['sales'][:90],
         label='Test (факт)', color='green', linewidth=1.5)
plt.plot(arima_predictions.index[:90], arima_predictions[:90],
         label='ARIMA прогноз', color='red', linewidth=1.5, linestyle='--')
plt.title(f'Прогноз ARIMA\nMAPE: {arima_mape:.2f}%', fontsize=11, fontweight='bold')
plt.xlabel('Дата')
plt.ylabel('Продажі')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

# 10.2. Прогноз ETS
ax11 = plt.subplot(4, 3, 11)
plt.plot(train_data.index[-90:], train_data['sales'][-90:],
         label='Train', color='blue', linewidth=1.5)
plt.plot(test_data.index[:90], test_data['sales'][:90],
         label='Test (факт)', color='green', linewidth=1.5)
plt.plot(ets_predictions.index[:90], ets_predictions[:90],
         label='ETS прогноз', color='orange', linewidth=1.5, linestyle='--')
plt.title(f'Прогноз ETS\nMAPE: {ets_mape:.2f}%', fontsize=11, fontweight='bold')
plt.xlabel('Дата')
plt.ylabel('Продажі')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

# 10.3. Прогноз Linear Regression
ax12 = plt.subplot(4, 3, 12)
plt.plot(train_data.index[-90:], train_data['sales'][-90:],
         label='Train', color='blue', linewidth=1.5)
plt.plot(y_test.index[:90], y_test[:90],
         label='Test (факт)', color='green', linewidth=1.5)
plt.plot(lr_pred_series.index[:90], lr_pred_series[:90],
         label='LR прогноз', color='purple', linewidth=1.5, linestyle='--')
plt.title(f'Прогноз Linear Regression\nMAPE: {lr_mape:.2f}%',
          fontsize=11, fontweight='bold')
plt.xlabel('Дата')
plt.ylabel('Продажі')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab15_time_series_results.png', dpi=300, bbox_inches='tight')
print("✓ Графіки збережено у файл 'lab15_time_series_results.png'")

# ============================================================================
# 11. ПОРІВНЯННЯ МОДЕЛЕЙ
# ============================================================================
print("\n11. ПОРІВНЯННЯ МОДЕЛЕЙ")
print("-" * 80)

# Створення таблиці порівняння
comparison_df = pd.DataFrame({
    'Модель': ['ARIMA(5,1,2)', 'ETS', 'Linear Regression'],
    'MAE': [arima_mae, ets_mae, lr_mae],
    'RMSE': [arima_rmse, ets_rmse, lr_rmse],
    'MAPE (%)': [arima_mape, ets_mape, lr_mape]
})

print("\nПорівняльна таблиця метрик:")
print(comparison_df.to_string(index=False))

# Найкраща модель
best_model_idx = comparison_df['MAPE (%)'].idxmin()
best_model = comparison_df.loc[best_model_idx, 'Модель']
best_mape = comparison_df.loc[best_model_idx, 'MAPE (%)']

print(f"\n✓ Найкраща модель: {best_model}")
print(f"  MAPE: {best_mape:.2f}%")

# Графік порівняння
fig2 = plt.figure(figsize=(14, 5))

# Порівняння метрик
ax1 = plt.subplot(1, 3, 1)
x = np.arange(len(comparison_df))
width = 0.25
plt.bar(x - width, comparison_df['MAE'], width, label='MAE', alpha=0.8)
plt.bar(x, comparison_df['RMSE'], width, label='RMSE', alpha=0.8)
plt.bar(x + width, comparison_df['MAPE (%)'], width, label='MAPE (%)', alpha=0.8)
plt.xlabel('Модель')
plt.ylabel('Значення')
plt.title('Порівняння метрик моделей', fontsize=12, fontweight='bold')
plt.xticks(x, comparison_df['Модель'], rotation=15, ha='right')
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Всі прогнози разом
ax2 = plt.subplot(1, 3, 2)
plt.plot(test_data.index[:90], test_data['sales'][:90],
         label='Фактичні', color='black', linewidth=2)
plt.plot(arima_predictions.index[:90], arima_predictions[:90],
         label='ARIMA', linewidth=1.5, linestyle='--', alpha=0.7)
plt.plot(ets_predictions.index[:90], ets_predictions[:90],
         label='ETS', linewidth=1.5, linestyle='--', alpha=0.7)
plt.plot(lr_pred_series.index[:90], lr_pred_series[:90],
         label='LR', linewidth=1.5, linestyle='--', alpha=0.7)
plt.title('Всі прогнози (перші 90 днів)', fontsize=12, fontweight='bold')
plt.xlabel('Дата')
plt.ylabel('Продажі')
plt.legend()
plt.grid(True, alpha=0.3)

# Помилки прогнозів
ax3 = plt.subplot(1, 3, 3)
arima_errors = test_data['sales'] - arima_predictions
ets_errors = test_data['sales'] - ets_predictions
lr_errors = y_test - lr_predictions

plt.boxplot([arima_errors.dropna(), ets_errors.dropna(), lr_errors.dropna()],
            labels=['ARIMA', 'ETS', 'LR'],
            patch_artist=True)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
plt.title('Розподіл помилок прогнозів', fontsize=12, fontweight='bold')
plt.ylabel('Помилка')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab15_model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Графік порівняння збережено у файл 'lab15_model_comparison.png'")

# ============================================================================
# 12. ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ
# ============================================================================
print("\n12. ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ")
print("-" * 80)

# Збереження прогнозів
predictions_df = pd.DataFrame({
    'date': test_data.index[:len(arima_predictions)],
    'actual': test_data['sales'][:len(arima_predictions)].values,
    'arima_forecast': arima_predictions.values,
    'ets_forecast': ets_predictions.values
})

predictions_df.to_csv('time_series_predictions.csv', index=False, encoding='utf-8-sig')
print("✓ Прогнози збережено у 'time_series_predictions.csv'")

# Збереження метрик
comparison_df.to_csv('model_metrics.csv', index=False, encoding='utf-8-sig')
print("✓ Метрики збережено у 'model_metrics.csv'")
