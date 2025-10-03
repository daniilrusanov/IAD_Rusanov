import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 1. Завантаження та підготовка даних ---
print("--- 1. Завантаження та підготовка даних ---")
filename = 'data.xlsx'
df = pd.read_excel(filename)
print(f"Файл '{filename}' успішно завантажено.\n")

# Обробка пропусків, кодування, масштабування...
for col in df.select_dtypes(include=np.number).columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())
for col in df.select_dtypes(exclude=np.number).columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

X = df.drop(['Група', 'Номер'], axis=1)
y = df['Група']
X_encoded = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Дані успішно підготовлено.\n")

output_dir = 'chart_lab5'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 2. Побудова та оцінка моделі Дерева рішень (CART) ---
print("=" * 60)
print("--- 2. Модель: Дерево рішень (CART) ---")
cart_model = DecisionTreeClassifier(random_state=42)
cart_model.fit(X_train_scaled, y_train)
y_pred_cart = cart_model.predict(X_test_scaled)
accuracy_cart = accuracy_score(y_test, y_pred_cart)
print(f"Точність (Accuracy) моделі CART: {accuracy_cart:.2%}")
print("\nЗвіт по класифікації:")
print(classification_report(y_test, y_pred_cart))

# --- Візуалізація дерева рішень ---
plt.figure(figsize=(20, 10))
plot_tree(cart_model,
          feature_names=X_encoded.columns,
          class_names=['Група 1', 'Група 2'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Візуалізація Дерева рішень (CART)", fontsize=16)
plt.savefig(os.path.join(output_dir, 'decision_tree.png'))
plt.close()
print("Графік дерева рішень збережено.")

# --- 3. Побудова та оцінка моделі KNN ---
print("=" * 60)
print("--- 3. Модель: K-найближчих сусідів (KNN) ---")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Точність (Accuracy) моделі KNN: {accuracy_knn:.2%}")
print("\nЗвіт по класифікації:")
print(classification_report(y_test, y_pred_knn))

# --- Візуалізація межі прийняття рішень для KNN ---
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
knn_model_pca = KNeighborsClassifier(n_neighbors=5).fit(X_train_pca, y_train)


def plot_decision_boundary(model, X, y, title, filename):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='viridis', s=100, edgecolor='k')
    plt.title(title, fontsize=16)
    plt.xlabel('Головна компонента 1')
    plt.ylabel('Головна компонента 2')
    plt.legend(title='Група')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


plot_decision_boundary(knn_model_pca, X_test_pca, y_test, 'Межа прийняття рішень (KNN, k=5)',
                       'decision_boundary_knn.png')
print("Графік межі прийняття рішень для KNN збережено.")

# --- 4. Порівняння результатів та висновки ---
print("=" * 60)
print("--- 4. Порівняння результатів та висновки ---")
print(f"Точність Дерева рішень (CART): {accuracy_cart:.2%}")
print(f"Точність K-найближчих сусідів (KNN): {accuracy_knn:.2%}\n")

if accuracy_cart > accuracy_knn:
    print("Висновок: Модель Дерева рішень (CART) показала кращу точність на цих даних.")
elif accuracy_knn > accuracy_cart:
    print("Висновок: Модель K-найближчих сусідів (KNN) показала кращу точність на цих даних.")
else:
    print("Висновок: Обидві моделі показали однакову точність.")
