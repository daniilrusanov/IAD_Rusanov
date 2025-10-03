import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as shc
import os

# --- 1. Завантаження та підготовка даних ---
print("--- 1. Завантаження та підготовка даних ---")
filename = 'data.xlsx'
try:
    df = pd.read_excel(filename)
    print(f"Файл '{filename}' успішно завантажено.\n")
except FileNotFoundError:
    print(f"Помилка: Файл '{filename}' не знайдено.")
    exit()

# Обробка пропусків, кодування...
for col in df.select_dtypes(include=np.number).columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())
for col in df.select_dtypes(exclude=np.number).columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# Для кластеризації ми використовуємо всі дані, крім службових стовпців
X = df.drop(['Група', 'Номер'], axis=1)
X_encoded = pd.get_dummies(X, drop_first=True)

# Масштабування ознак
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
print("Дані успішно підготовлено та масштабовано.\n")

# Створення директорії для збереження графіків
output_dir = 'chart_lab8'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 2. Ієрархічна агломеративна кластеризація ---
print("="*60)
print("--- 2. Ієрархічна кластеризація та дендрограма ---")

plt.figure(figsize=(12, 8))
plt.title("Дендрограма ієрархічної кластеризації")
# Використовуємо метод Уорда (Ward's method) для мінімізації дисперсії всередині кластерів
dend = shc.dendrogram(shc.linkage(X_scaled, method='ward'))
plt.axhline(y=12, color='r', linestyle='--') # Горизонтальна лінія для визначення кількості кластерів
plt.ylabel("Відстань (Ward's distance)")
plt.savefig(os.path.join(output_dir, 'dendrogram.png'))
plt.close()
print("Дендрограму побудовано та збережено.")


# --- 3. Визначення оптимального K для K-Means (Метод ліктя) ---
print("\n" + "="*60)
print("--- 3. Визначення оптимальної кількості кластерів (Метод ліктя) ---")
wcss = [] # Within-Cluster Sum of Squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Метод ліктя для визначення оптимального K')
plt.xlabel('Кількість кластерів (K)')
plt.ylabel('WCSS (Інерція)')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'elbow_method.png'))
plt.close()
print("Графік 'методу ліктя' збережено.")

linkage_matrix = shc.linkage(X_scaled, method='ward')
print("Схема об'єднання (перші 5 кроків):")
print(" [кластер 1, кластер 2, відстань, к-ть об'єктів]")
print(linkage_matrix[:5])


# --- 4. Побудова моделі K-Means та візуалізація ---
print("="*60)
print("--- 4. Побудова фінальної моделі K-Means (K=2) та візуалізація ---")
optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Для візуалізації зменшуємо розмірність до 2-х компонент
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Створюємо DataFrame для зручності візуалізації
pca_df = pd.DataFrame(data=X_pca, columns=['Головна компонента 1', 'Головна компонента 2'])
pca_df['Кластер'] = clusters
centers_pca = pca.transform(kmeans.cluster_centers_)

# Візуалізація
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Головна компонента 1', y='Головна компонента 2', hue='Кластер', data=pca_df, palette='viridis', s=100, alpha=0.8)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], s=300, c='red', marker='X', label='Центроїди')
plt.title('Кластеризація K-Means (K=2) у просторі головних компонент')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'kmeans_clusters.png'))
plt.close()
print("Графік з результатами кластеризації K-Means збережено.")

centroids_df = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_encoded.columns)
print("\nКоординати центрів кластерів (профіль 'середнього' користувача):")
print(centroids_df.round(2))