import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("--- 1. Завантаження та підготовка даних ---")
filename = 'data.xlsx'
try:
    df = pd.read_excel(filename)
    print(f"Файл '{filename}' успішно завантажено.\n")
except FileNotFoundError:
    print(f"Помилка: Файл '{filename}' не знайдено.")
    exit()

if 'Номер' in df.columns:
    df = df.drop('Номер', axis=1)
group_labels = df['Група'].copy()
df_processed = pd.get_dummies(df, drop_first=True)
if 'Група' in df_processed.columns:
    df_processed = df_processed.drop('Група', axis=1)

print("-- Датасет після перетворення категоріальних ознак (до масштабування) --")
print(df_processed.head())
print("\n" + "="*60 + "\n")


print("--- 2. Масштабування числових ознак ---")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df_processed)
df_scaled = pd.DataFrame(features_scaled, columns=df_processed.columns)
print("Дані успішно масштабовано за допомогою StandardScaler.")
print("-- Перші 5 рядків даних ПІСЛЯ масштабування --")
print(df_scaled.head())
print("\n" + "="*60 + "\n")


print("--- 3. Застосування методу головних компонент (PCA) ---")
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features_scaled)

pca_df = pd.DataFrame(data=principal_components,
                      columns=['Головна компонента 1', 'Головна компонента 2'])
pca_df['Група'] = group_labels

print("PCA застосовано. Розмірність зменшено до 2 компонент.")
print("-- Перші 5 рядків даних ПІСЛЯ PCA (у новому просторі компонент) --")
print(pca_df.head())
print("\n" + "="*60 + "\n")


print("--- 4. Аналіз частки пояснюваної дисперсії ---")
explained_variance = pca.explained_variance_ratio_
print(f"Частка дисперсії, пояснена першою компонентою: {explained_variance[0]:.2%}")
print(f"Частка дисперсії, пояснена другою компонентою: {explained_variance[1]:.2%}")
print(f"Загальна частка дисперсії, пояснена двома компонентами: {np.sum(explained_variance):.2%}\n")


print("--- 5. Візуалізація даних у просторі головних компонент ---")
output_dir = 'chart_lab4'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='Головна компонента 1',
    y='Головна компонента 2',
    hue='Група',
    data=pca_df,
    palette='viridis',
    s=100,
    alpha=0.8
)
plt.title('Візуалізація даних у просторі 2-х головних компонент')
plt.xlabel('Головна компонента 1')
plt.ylabel('Головна компонента 2')
plt.grid(True)
plt.legend(title='Група')
chart_path = os.path.join(output_dir, 'pca_visualization.png')
plt.savefig(chart_path)
print(f"Графік успішно збережено у файл: {chart_path}")
