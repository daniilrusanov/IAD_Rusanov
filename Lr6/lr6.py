import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

# Обробка пропусків
for col in df.select_dtypes(include=np.number).columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())
for col in df.select_dtypes(exclude=np.number).columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# Визначення цільової змінної (y) та ознак (X)
X = df.drop(['Група', 'Номер'], axis=1)
y = df['Група']

# Кодування та масштабування
X_encoded = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Дані успішно підготовлено та розділено на тренувальний і тестовий набори.\n")

# Створення директорії для збереження графіків
output_dir = 'chart_lab'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 2. Побудова моделі Naive Bayes ---
print("="*60)
print("--- 2. Модель: Naive Bayes (Гаусівський наївний баєсів класифікатор) ---")
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
y_pred_nb = nb_model.predict(X_test_scaled)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Точність (Accuracy) моделі Naive Bayes: {accuracy_nb:.2%}")
print("\nЗвіт по класифікації:")
print(classification_report(y_test, y_pred_nb))

# --- 3. Побудова моделі SVM ---
print("="*60)
print("--- 3. Модель: Support Vector Machine (Метод опорних векторів) ---")
svm_model = SVC(kernel='linear', random_state=42) # Використовуємо лінійне ядро для простоти
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Точність (Accuracy) моделі SVM: {accuracy_svm:.2%}")
print("\nЗвіт по класифікації:")
print(classification_report(y_test, y_pred_svm))

# --- 4. Візуалізація результатів ---
print("="*60)
print("--- 4. Візуалізація результатів ---")

# 4.1. Матриці похибок
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# Naive Bayes
cm_nb = confusion_matrix(y_test, y_pred_nb)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Матриця похибок для Naive Bayes')
axes[0].set_xlabel('Передбачений клас')
axes[0].set_ylabel('Істинний клас')
# SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Матриця похибок для SVM')
axes[1].set_xlabel('Передбачений клас')
axes[1].set_ylabel('Істинний клас')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'))
print("Матриці похибок збережено.")
plt.close()

# 4.2. Графіки розподілу ключових ознак
print("Графіки розподілу даних збережено.")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.histplot(data=df, x='К-ть годин', hue='Група', kde=True, palette='viridis', ax=axes[0])
axes[0].set_title('Розподіл за кількістю годин на тиждень')
sns.histplot(data=df, x='Стаж', hue='Група', kde=True, palette='viridis', ax=axes[1])
axes[1].set_title('Розподіл за стажем роботи в мережі')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
plt.close()


# --- 5. Порівняння та висновки ---
print("="*60)
print("--- 5. Порівняння результатів та висновки ---")
print(f"Точність Naive Bayes: {accuracy_nb:.2%}")
print(f"Точність SVM: {accuracy_svm:.2%}\n")
