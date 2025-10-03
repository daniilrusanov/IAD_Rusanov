import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
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

# Обробка пропусків...
for col in df.select_dtypes(include='number').columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())
for col in df.select_dtypes(exclude='number').columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# Перетворення даних у транзакційний формат
# Дискретизація числових ознак (розбиття на категорії)
df_trans = pd.DataFrame()
df_trans['Вік'] = pd.qcut(df['Вік'], q=3, labels=['Молодий', 'Середній', 'Старший'])
df_trans['Стаж'] = pd.qcut(df['Стаж'], q=3, labels=['Новачок', 'Досвідчений', 'Експерт'])
df_trans['К-ть годин'] = pd.qcut(df['К-ть годин'], q=2, labels=['Мало годин', 'Багато годин'])
df_trans['Дохід'] = pd.qcut(df['Дохід'], q=3, labels=['Низький', 'Середній', 'Високий'])
df_trans['Професія'] = 'Професія_' + df['Професія']
df_trans['Стать'] = 'Стать_' + df['Стать']

# Створення списку транзакцій
transactions = df_trans.apply(lambda x: x.dropna().tolist(), axis=1).tolist()
print("Приклад перших 5 транзакцій:")
print(*transactions[:5], sep="\n")

# Кодування транзакцій у формат, придатний для Apriori
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
print("\nДані успішно перетворено у транзакційний формат.\n")

output_dir = 'chart_lab9'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 2. Виявлення частих наборів (Apriori) ---
print("="*60)
print("--- 2. Виявлення частих наборів елементів (min_support = 0.2) ---")
frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)
print(frequent_itemsets.sort_values(by='support', ascending=False).head(10))

# --- 3. Побудова асоціативних правил ---
print("\n" + "="*60)
print("--- 3. Побудова асоціативних правил (min_confidence = 0.7) ---")
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
# Виводимо найцікавіші правила, відсортовані за lift (показує силу зв'язку)
print(rules.sort_values(by='lift', ascending=False).head(10))

# --- 4. Побудова графа правил ---
print("\n" + "="*60)
print("--- 4. Побудова графа правил ---")
plt.figure(figsize=(12, 12))
G = nx.from_pandas_edgelist(rules, 'antecedents', 'consequents', ['lift'])
# Перетворення frozenset на рядки для підписів
labels = {node: ', '.join(node) for node in G.nodes()}
nx.draw(G, with_labels=True, labels=labels, node_size=2000, node_color='skyblue', font_size=10, width=[d['lift'] for (u, v, d) in G.edges(data=True)])
plt.title("Граф асоціативних правил (товщина ребра = lift)")
plt.savefig(os.path.join(output_dir, 'rules_graph.png'))
plt.close()
print("Граф правил збережено.")

# --- 5 & 6. Матриця частот та підтримки ---
print("\n" + "="*60)
print("--- 5-6. Матриця підтримки для парних наборів ---")
# Фільтруємо набори, що містять 2 елементи
itemsets_pairs = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == 2)].copy()
# Створюємо стовпці для першого та другого елемента
itemsets_pairs[['item1', 'item2']] = pd.DataFrame(itemsets_pairs['itemsets'].tolist(), index=itemsets_pairs.index)
# Будуємо матрицю
support_matrix = itemsets_pairs.pivot(index='item1', columns='item2', values='support').fillna(0)

plt.figure(figsize=(12, 10))
sns.heatmap(support_matrix, annot=True, cmap='viridis')
plt.title("Матриця підтримки (Support) для пар елементів")
plt.ylabel("Перший елемент")
plt.xlabel("Другий елемент")
plt.savefig(os.path.join(output_dir, 'support_matrix.png'))
plt.close()
print("Матрицю підтримки збережено.")

# --- 7. Візуалізація правил ---
print("\n" + "="*60)
print("--- 7. Візуалізація правил (Support vs. Confidence) ---")
plt.figure(figsize=(12, 8))
sns.scatterplot(x=rules['support'], y=rules['confidence'], size=rules['lift'], hue=rules['lift'], palette='viridis', sizes=(20, 500))
plt.title('Асоціативні правила: Support vs Confidence (розмір = lift)')
plt.xlabel('Підтримка (Support)')
plt.ylabel('Впевненість (Confidence)')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'rules_scatter_plot.png'))
plt.close()
print("Графік візуалізації правил збережено.")