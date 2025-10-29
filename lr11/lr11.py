import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, precision_score, recall_score)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Завантаження необхідних ресурсів NLTK
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# ============================================================================
# 1. ЗАВАНТАЖЕННЯ ДАТАСЕТУ
# ============================================================================
print("\n1. ЗАВАНТАЖЕННЯ ДАТАСЕТУ")
print("-" * 80)

# Вибираємо 4 категорії для класифікації
categories = ['alt.atheism', 'sci.space', 'comp.graphics', 'rec.sport.baseball']

# Завантаження тренувальних та тестових даних
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories,
                                     shuffle=True, random_state=42)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories,
                                    shuffle=True, random_state=42)

print(f"Категорії: {newsgroups_train.target_names}")
print(f"Кількість тренувальних документів: {len(newsgroups_train.data)}")
print(f"Кількість тестових документів: {len(newsgroups_test.data)}")

# Розподіл по категоріях
train_counts = pd.Series(newsgroups_train.target).value_counts().sort_index()
print("\nРозподіл тренувальних даних:")
for idx, count in train_counts.items():
    print(f"  {newsgroups_train.target_names[idx]}: {count}")

# ============================================================================
# 2. ПОПЕРЕДНЯ ОБРОБКА ТЕКСТУ
# ============================================================================
print("\n2. ПОПЕРЕДНЯ ОБРОБКА ТЕКСТУ")
print("-" * 80)

# Приклад оригінального тексту
print("Приклад оригінального тексту:")
print(newsgroups_train.data[0][:300])
print("...")

# Функція для попередньої обробки
def preprocess_text(text):
    """Попередня обробка тексту"""
    # Приведення до нижнього регістру
    text = text.lower()
    # Токенізація
    tokens = word_tokenize(text)
    # Видалення стоп-слів та коротких слів
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and
              word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

# Обробка перших кількох документів для демонстрації
print("\nПриклад обробленого тексту:")
processed_sample = preprocess_text(newsgroups_train.data[0])
print(processed_sample[:300])
print("...")

# ============================================================================
# 3. ВЕКТОРИЗАЦІЯ TF-IDF
# ============================================================================
print("\n3. ВЕКТОРИЗАЦІЯ ТЕКСТІВ")
print("-" * 80)

# TF-IDF векторизація
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2),
    stop_words='english'
)

X_train_tfidf = tfidf_vectorizer.fit_transform(newsgroups_train.data)
X_test_tfidf = tfidf_vectorizer.transform(newsgroups_test.data)

print(f"Розмір TF-IDF матриці (train): {X_train_tfidf.shape}")
print(f"Розмір TF-IDF матриці (test): {X_test_tfidf.shape}")
print(f"Кількість унікальних термінів: {len(tfidf_vectorizer.get_feature_names_out())}")

# Для порівняння - звичайний Bag of Words
count_vectorizer = CountVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2),
    stop_words='english'
)

X_train_count = count_vectorizer.fit_transform(newsgroups_train.data)
X_test_count = count_vectorizer.transform(newsgroups_test.data)

print(f"\nРозмір Count матриці (train): {X_train_count.shape}")

# Приклад найважливіших слів за TF-IDF для першого документа
print("\nТоп-10 слів з найвищим TF-IDF для першого документа:")
feature_names = tfidf_vectorizer.get_feature_names_out()
doc_vector = X_train_tfidf[0].toarray()[0]
top_indices = doc_vector.argsort()[-10:][::-1]
for idx in top_indices:
    if doc_vector[idx] > 0:
        print(f"  {feature_names[idx]}: {doc_vector[idx]:.4f}")

# ============================================================================
# 4. ТРЕНУВАННЯ МОДЕЛЕЙ КЛАСИФІКАЦІЇ
# ============================================================================
print("\n4. ТРЕНУВАННЯ МОДЕЛЕЙ КЛАСИФІКАЦІЇ")
print("-" * 80)

y_train = newsgroups_train.target
y_test = newsgroups_test.target

# Словник для зберігання результатів
results = {}

# 4.1. Naive Bayes з TF-IDF
print("\n4.1. Навчання Naive Bayes (TF-IDF)...")
nb_tfidf = MultinomialNB(alpha=0.1)
nb_tfidf.fit(X_train_tfidf, y_train)
y_pred_nb_tfidf = nb_tfidf.predict(X_test_tfidf)
results['Naive Bayes (TF-IDF)'] = {
    'accuracy': accuracy_score(y_test, y_pred_nb_tfidf),
    'f1': f1_score(y_test, y_pred_nb_tfidf, average='weighted'),
    'precision': precision_score(y_test, y_pred_nb_tfidf, average='weighted'),
    'recall': recall_score(y_test, y_pred_nb_tfidf, average='weighted'),
    'predictions': y_pred_nb_tfidf
}

# 4.2. Naive Bayes з Count
print("4.2. Навчання Naive Bayes (Count)...")
nb_count = MultinomialNB(alpha=0.1)
nb_count.fit(X_train_count, y_train)
y_pred_nb_count = nb_count.predict(X_test_count)
results['Naive Bayes (Count)'] = {
    'accuracy': accuracy_score(y_test, y_pred_nb_count),
    'f1': f1_score(y_test, y_pred_nb_count, average='weighted'),
    'precision': precision_score(y_test, y_pred_nb_count, average='weighted'),
    'recall': recall_score(y_test, y_pred_nb_count, average='weighted'),
    'predictions': y_pred_nb_count
}

# 4.3. SVM з TF-IDF
print("4.3. Навчання SVM (TF-IDF)...")
svm_tfidf = LinearSVC(random_state=42, max_iter=1000)
svm_tfidf.fit(X_train_tfidf, y_train)
y_pred_svm_tfidf = svm_tfidf.predict(X_test_tfidf)
results['SVM (TF-IDF)'] = {
    'accuracy': accuracy_score(y_test, y_pred_svm_tfidf),
    'f1': f1_score(y_test, y_pred_svm_tfidf, average='weighted'),
    'precision': precision_score(y_test, y_pred_svm_tfidf, average='weighted'),
    'recall': recall_score(y_test, y_pred_svm_tfidf, average='weighted'),
    'predictions': y_pred_svm_tfidf
}

# 4.4. SVM з Count
print("4.4. Навчання SVM (Count)...")
svm_count = LinearSVC(random_state=42, max_iter=1000)
svm_count.fit(X_train_count, y_train)
y_pred_svm_count = svm_count.predict(X_test_count)
results['SVM (Count)'] = {
    'accuracy': accuracy_score(y_test, y_pred_svm_count),
    'f1': f1_score(y_test, y_pred_svm_count, average='weighted'),
    'precision': precision_score(y_test, y_pred_svm_count, average='weighted'),
    'recall': recall_score(y_test, y_pred_svm_count, average='weighted'),
    'predictions': y_pred_svm_count
}

# 4.5. Logistic Regression з TF-IDF
print("4.5. Навчання Logistic Regression (TF-IDF)...")
lr_tfidf = LogisticRegression(random_state=42, max_iter=1000)
lr_tfidf.fit(X_train_tfidf, y_train)
y_pred_lr_tfidf = lr_tfidf.predict(X_test_tfidf)
results['Logistic Regression (TF-IDF)'] = {
    'accuracy': accuracy_score(y_test, y_pred_lr_tfidf),
    'f1': f1_score(y_test, y_pred_lr_tfidf, average='weighted'),
    'precision': precision_score(y_test, y_pred_lr_tfidf, average='weighted'),
    'recall': recall_score(y_test, y_pred_lr_tfidf, average='weighted'),
    'predictions': y_pred_lr_tfidf
}

# ============================================================================
# 5. ОЦІНКА ТА ПОРІВНЯННЯ МОДЕЛЕЙ
# ============================================================================
print("\n5. РЕЗУЛЬТАТИ КЛАСИФІКАЦІЇ")
print("-" * 80)

# Таблиця результатів
results_df = pd.DataFrame(results).T
results_df = results_df[['accuracy', 'precision', 'recall', 'f1']]
results_df = results_df.round(4)
print("\nПорівняння моделей:")
print(results_df)

# Найкраща модель
best_model = results_df['f1'].idxmax()
print(f"\n✓ Найкраща модель: {best_model}")
print(f"  F1-Score: {results_df.loc[best_model, 'f1']:.4f}")
print(f"  Accuracy: {results_df.loc[best_model, 'accuracy']:.4f}")

# ============================================================================
# 6. ДЕТАЛЬНИЙ АНАЛІЗ НАЙКРАЩОЇ МОДЕЛІ
# ============================================================================
print("\n6. ДЕТАЛЬНИЙ АНАЛІЗ НАЙКРАЩОЇ МОДЕЛІ")
print("-" * 80)

best_predictions = results[best_model]['predictions']

print(f"\nКласифікаційний звіт для {best_model}:")
print(classification_report(y_test, best_predictions,
                          target_names=newsgroups_train.target_names))

# Аналіз по категоріях
print("\nМетрики по категоріях:")
for i, category in enumerate(newsgroups_train.target_names):
    mask = y_test == i
    acc = accuracy_score(y_test[mask], best_predictions[mask])
    print(f"  {category}: {acc:.4f}")

# ============================================================================
# 7. ВІЗУАЛІЗАЦІЯ
# ============================================================================
print("\n7. СТВОРЕННЯ ВІЗУАЛІЗАЦІЙ")
print("-" * 80)

fig = plt.figure(figsize=(16, 12))

# 7.1. Порівняння моделей
ax1 = plt.subplot(2, 3, 1)
results_df.plot(kind='bar', ax=ax1, width=0.8)
plt.title('Порівняння метрик різних моделей', fontsize=12, fontweight='bold')
plt.ylabel('Значення')
plt.xlabel('Модель')
plt.xticks(rotation=45, ha='right')
plt.legend(loc='lower right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# 7.2. Порівняння TF-IDF vs Count
ax2 = plt.subplot(2, 3, 2)
comparison_data = {
    'TF-IDF': [results['Naive Bayes (TF-IDF)']['f1'],
               results['SVM (TF-IDF)']['f1']],
    'Count': [results['Naive Bayes (Count)']['f1'],
              results['SVM (Count)']['f1']]
}
x = np.arange(2)
width = 0.35
plt.bar(x - width/2, comparison_data['TF-IDF'], width, label='TF-IDF', alpha=0.8)
plt.bar(x + width/2, comparison_data['Count'], width, label='Count', alpha=0.8)
plt.xlabel('Алгоритм')
plt.ylabel('F1-Score')
plt.title('TF-IDF vs Count Vectorization', fontsize=12, fontweight='bold')
plt.xticks(x, ['Naive Bayes', 'SVM'])
plt.legend()
plt.grid(axis='y', alpha=0.3)

# 7.3. Матриця помилок
ax3 = plt.subplot(2, 3, 3)
cm = confusion_matrix(y_test, best_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[name.split('.')[-1] for name in newsgroups_train.target_names],
            yticklabels=[name.split('.')[-1] for name in newsgroups_train.target_names])
plt.title(f'Матриця помилок\n{best_model}', fontsize=12, fontweight='bold')
plt.ylabel('Справжній клас')
plt.xlabel('Передбачений клас')

# 7.4. F1-Score по категоріях
ax4 = plt.subplot(2, 3, 4)
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1_per_class, support = precision_recall_fscore_support(
    y_test, best_predictions, average=None
)

colors = plt.cm.viridis(np.linspace(0, 1, len(newsgroups_train.target_names)))
bars = plt.bar(range(len(newsgroups_train.target_names)), f1_per_class, color=colors, alpha=0.8)
plt.xlabel('Категорія')
plt.ylabel('F1-Score')
plt.title('F1-Score по категоріях', fontsize=12, fontweight='bold')
plt.xticks(range(len(newsgroups_train.target_names)),
           [name.split('.')[-1] for name in newsgroups_train.target_names],
           rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Додаємо значення на стовпчики
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 7.5. Розподіл документів по категоріях
ax5 = plt.subplot(2, 3, 5)
category_counts = [np.sum(y_test == i) for i in range(len(newsgroups_train.target_names))]
plt.pie(category_counts, labels=[name.split('.')[-1] for name in newsgroups_train.target_names],
        autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Розподіл тестових даних', fontsize=12, fontweight='bold')

# 7.6. Топ слів за TF-IDF для кожної категорії
ax6 = plt.subplot(2, 3, 6)
# Отримуємо середні TF-IDF значення для кожної категорії
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
top_words_per_category = []

for i in range(len(newsgroups_train.target_names)):
    # Знаходимо документи цієї категорії
    mask = y_train == i
    # Обчислюємо середні TF-IDF
    avg_tfidf = np.asarray(X_train_tfidf[mask].mean(axis=0)).ravel()
    # Топ-5 слів
    top_indices = avg_tfidf.argsort()[-5:][::-1]
    top_words = feature_names[top_indices]
    top_words_per_category.append(', '.join(top_words))

plt.axis('off')
table_data = [[newsgroups_train.target_names[i].split('.')[-1], top_words_per_category[i]]
              for i in range(len(newsgroups_train.target_names))]
table = plt.table(cellText=table_data, colLabels=['Категорія', 'Топ-5 слів (TF-IDF)'],
                 cellLoc='left', loc='center', colWidths=[0.3, 0.7])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
plt.title('Найважливіші слова для кожної категорії', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('lab11_results.png', dpi=300, bbox_inches='tight')
print("Графіки збережено у файл 'lab11_results.png'")