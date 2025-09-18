from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

filename = 'data.xlsx'
6
try:
    # --- 1. Завантажити датасет ---
    df = pd.read_excel(filename)
    print("--- 1. Завантажити датасет з пропущеними значеннями. ---")
    print(f"Файл '{filename}' успішно завантажено.\n")
    print("-- Початковий вигляд даних ---")
    print(df.head())
    print("\n--- Загальна інформація про стовпці та типи даних ---")
    df.info()

    # --- 2. Виявити та обробити пропущені дані ---
    print("\n--- 2. Виявити та обробити пропущені дані. ---")
    print("Кількість пропусків до обробки:")
    print(df.isnull().sum())

    if df.isnull().sum().sum() > 0:
        # Визначаємо числові та категоріальні стовпці
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        # Заповнюємо пропуски для числових стовпців середнім значенням
        for col in numerical_cols:
            if df[col].isnull().any():
                mean_value = df[col].mean()
                df[col] = df[col].fillna(mean_value)

        # Заповнюємо пропуски для категоріальних стовпців модою
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_value = df[col].mode()[0]
                df[col] = df[col].fillna(mode_value)

        print("\nПропуски заповнено:")
        print(df.isnull().sum())
    else:
        print("Пропущені дані відсутні.")

    # --- 3. Видалити дублікати ---
    print("\n--- 3. Видалити дубликати. ---")
    duplicate_count = df.duplicated().sum()
    print(f"Кількість дублікатів: {duplicate_count}")

    if duplicate_count > 0:
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        print("Дублікати видалено.")
        print(f"Новий розмір датасету: {df.shape}")

    # Визначаємо типи стовпців знову, оскільки вони нам потрібні для наступних кроків
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # --- 4. Закодувати категоріальні змінні ---
    print("\n--- 4. Закодувати категоріальні змінні. ---")
    if categorical_cols:
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        print("Дата після кодування категоріальних ознак:")
        print(df_encoded.head())
        print(f"Розмір датасету після кодування: {df_encoded.shape}")
    else:
        df_encoded = df.copy()
        print("Категоріальні стовпці відсутні.")

    # --- 5. Масштабувати числові ознаки ---
    print("\n--- 5. Масштабувати числові ознаки. ---")
    if numerical_cols:
        scaler = StandardScaler()
        cols_to_scale = [col for col in numerical_cols if col in df_encoded.columns]
        df_encoded[cols_to_scale] = scaler.fit_transform(df_encoded[cols_to_scale])
        print("\nКінцева версія датасету:")
        print(df_encoded.head())
    else:
        print("Числові стовпці відсутні.")

except FileNotFoundError:
    print(f"Помилка: Файл '{filename}' не знайдено. Переконайтесь, що він знаходиться в одній папці зі скриптом.")
except Exception as e:
    print(f"Сталася неочікувана помилка: {e}")