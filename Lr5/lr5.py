import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("--- 1. Завантаження та підготовка даних ---")
filename = 'data.xlsx'
try:
    df = pd.read_excel(filename)
    print(f"Файл '{filename}' успішно завантажено.\n")
except FileNotFoundError:
    print(f"Помилка: Файл '{filename}' не знайдено.")
    exit()

# Обробка пропускі
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
for col in numerical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())
for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# Визначення цільової змінної (y) та ознак (X)
X = df.drop(['Група', 'Номер'], axis=1)
y = df['Група']

# Кодування категоріальних ознак
X_encoded = pd.get_dummies(X, drop_first=True)
print("-- Ознаки після кодування (перші 5 рядків) --")
print(X_encoded.head())
print("\n")

# Розділення даних на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42, stratify=y)
print(f"Розмір тренувального набору: {X_train.shape}")
print(f"Розмір тестового набору: {X_test.shape}\n")

# Масштабування числових ознак
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- 2. Побудова та оцінка моделі Дерева рішень (CART) ---
print("="*60)
print("--- 2. Модель: Дерево рішень (CART) ---")
# Ініціалізація та навчання моделі
cart_model = DecisionTreeClassifier(random_state=42)
cart_model.fit(X_train_scaled, y_train)

# Прогнозування на тестових даних
y_pred_cart = cart_model.predict(X_test_scaled)

# Оцінка моделі
accuracy_cart = accuracy_score(y_test, y_pred_cart)
print(f"Точність (Accuracy) моделі CART: {accuracy_cart:.2%}")
print("\nЗвіт по класифікації (Classification Report):")
print(classification_report(y_test, y_pred_cart))


# --- 3. Побудова та оцінка моделі KNN ---
print("="*60)
print("--- 3. Модель: K-найближчих сусідів (KNN) ---")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
# Прогнозування на тестових даних
y_pred_knn = knn_model.predict(X_test_scaled)
# Оцінка моделі
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Точність (Accuracy) моделі KNN: {accuracy_knn:.2%}")
print("\nЗвіт по класифікації (Classification Report):")
print(classification_report(y_test, y_pred_knn))


# --- 4. Порівняння результатів та висновки ---
print("="*60)
print("--- 4. Порівняння результатів та висновки ---")
print(f"Точність Дерева рішень (CART): {accuracy_cart:.2%}")
print(f"Точність K-найближчих сусідів (KNN): {accuracy_knn:.2%}\n")

if accuracy_cart > accuracy_knn:
    print("Висновок: Модель Дерева рішень (CART) показала кращу точність на цих даних.")
elif accuracy_knn > accuracy_cart:
    print("Висновок: Модель K-найближчих сусідів (KNN) показала кращу точність на цих даних.")
else:
    print("Висновок: Обидві моделі показали однакову точність.")