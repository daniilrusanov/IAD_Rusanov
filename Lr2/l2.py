import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

filename = 'data.xlsx'

try:
    df = pd.read_excel(filename)
    print("--- 1. Завантажити датасет ---")
    print(f"Файл '{filename}' успішно завантажено.\n")
    print("-- Початковий вигляд даних ---")
    print(df.head())
    print("\n--- Загальна інформація про стовпці та типи даних ---")
    df.info()
except FileNotFoundError:
    print(f"Помилка '{filename}' не знайдено")

numerical = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Номер' in numerical:
    numerical.remove('Номер')
categorical = df.select_dtypes(include='object').columns.tolist()
print(f"Числові ознаки для аналізу: {numerical}")
print(f"Категоріальні ознаки для аналізу: {categorical}")

print(f"\n\n--- 2. Розрахунок основних статистик ---")
print(f"--- Загальна статистика ---")
print(df[numerical].describe())

print(f"\n\n--- 3. Детальна статистика для кожної числової ознаки ---")
for col in numerical:
    print(f"--- {col} ---")
    print(f" - Математичне сподівання: {df[col].mean():.4f}")
    print(f" - Оцінка медіани: {df[col].median():.4f}")
    print(f" - Оцінка дисперсії: {df[col].var():.4f}")
    print(f" - Оцінка середньоквадратичного відхилення: {df[col].std():.4f}")
    print(f" - Оцінка коефіцієнта асиметрії: {df[col].skew():.4f}")
    print(f" - Оцінка коефіцієнту ексцесу: {df[col].kurtosis():.4f}")
    print(f" - Мінімальне значення: {df[col].min()}")
    print(f" - Максимальне значення: {df[col].max()}\n")

print(f"\n--- 4. Побудова графіків ---")
if not os.path.exists('chart_lab2'):
    os.makedirs('chart_lab2')

print("\nПобудова категоризованих гістограм")
plt.figure(figsize=(12, 7))
sns.histplot(data=df, x='Вік', hue='Професія', multiple='stack', kde=True)
plt.title('Розподіл віку за професіями')
plt.savefig('chart_lab2/histogram_age_by_profession.png')
plt.close()
print("Графік збережено!")

print("\nПобудова радіальної діаграми")
radar_df = df.groupby('Група')[numerical].mean()
labels = radar_df.columns
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fix, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for i, row in radar_df.iterrows():
    data = row.tolist()
    data += data[:1]
    ax.plot(angles, data, label=f"Група {i}")
    ax.fill(angles, data, alpha=0.2)

plt.xticks(angles[:-1], labels)
ax.set_title("Порівняння середніх показників по групах")
plt.legend()
plt.savefig('chart_lab2/radar_chart_by_group.png')
plt.close()
print("Графік збережено!")

print("\nПобудова гістограм boxplot")
for col in numerical:
    fix, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.histplot(df[col], kde=True, ax=axes[0])
    axes[0].set_title(f'Гістограма для "{col}"')
    sns.boxplot(x=df[col], ax=axes[1])
    axes[1].set_title(f'Boxplot для "{col}"')
    plt.savefig(f'chart_lab2/{col}.png')
    plt.close()
print("Графіки збережено!")

print("\nПобудова scatter plot")
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x='Стаж', y='Дохід', hue='Стать', style='Група', s=80)
plt.title("Залежність Доходу від Стажу")
plt.grid(True)
plt.savefig('chart_lab2/scatter_стаж_vs_дохід.png')
plt.close()
print("Графік збережено!")

print("\nПобудова pairplot plot")
pairplot = sns.pairplot(df[numerical + ['Стать']], hue='Стать')
pairplot.savefig('chart_lab2/pairplot.png')
plt.close()
print("Графік збережено!")
