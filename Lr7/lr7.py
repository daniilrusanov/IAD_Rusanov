import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
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
print("Дані успішно підготовлено та розділено.\n")

# --- 2. Побудова моделі Random Forest ---
print("=" * 60)
print("--- 2. Модель: Random Forest (Випадковий ліс) ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Точність (Accuracy) моделі Random Forest: {accuracy_rf:.2%}")
print("\nЗвіт по класифікації:")
print(classification_report(y_test, y_pred_rf))

# --- 3. Побудова моделі Gradient Boosting ---
print("=" * 60)
print("--- 3. Модель: Gradient Boosting (Градієнтний бустинг) ---")
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train_scaled, y_train)
y_pred_gb = gb_model.predict(X_test_scaled)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"Точність (Accuracy) моделі Gradient Boosting: {accuracy_gb:.2%}")
print("\nЗвіт по класифікації:")
print(classification_report(y_test, y_pred_gb))

# --- 4. Візуалізація результатів ---
print("=" * 60)
print("--- 4. Візуалізація результатів ---")
output_dir = 'chart_lab7'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
features = X_encoded.columns

# Графік для Random Forest
importance_rf_df = pd.DataFrame({'Feature': features, 'Importance': rf_model.feature_importances_}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_rf_df, hue='Feature', palette='viridis', legend=False)
plt.title('Важливість ознак для моделі Random Forest', fontsize=16)
plt.xlabel('Важливість')
plt.ylabel('Ознака')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance_rf.png'))
plt.close()
print("Графік важливості ознак для Random Forest збережено.")

# Графік для Gradient Boosting
importance_gb_df = pd.DataFrame({'Feature': features, 'Importance': gb_model.feature_importances_}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_gb_df, hue='Feature', palette='plasma', legend=False)
plt.title('Важливість ознак для моделі Gradient Boosting', fontsize=16)
plt.xlabel('Важливість')
plt.ylabel('Ознака')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance_gb.png'))
plt.close()
print("Графік важливості ознак для Gradient Boosting збережено.")


# --- 4.2 Візуалізація межі прийняття рішень (на основі PCA) ---
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
rf_model_pca = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_pca, y_train)
gb_model_pca = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_train_pca, y_train)

def plot_decision_boundary(model, X, y, title, filename):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='viridis', s=100, edgecolor='k')
    plt.title(title)
    plt.xlabel('Головна компонента 1')
    plt.ylabel('Головна компонента 2')
    plt.legend(title='Група')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

plot_decision_boundary(rf_model_pca, X_test_pca, y_test, 'Межа прийняття рішень (Random Forest)', 'decision_boundary_rf.png')
plot_decision_boundary(gb_model_pca, X_test_pca, y_test, 'Межа прийняття рішень (Gradient Boosting)', 'decision_boundary_gb.png')
print("Графіки меж прийняття рішень для обох моделей збережено.")