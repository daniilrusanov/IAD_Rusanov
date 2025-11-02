import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Налаштування matplotlib для української мови
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 1. ГЕНЕРАЦІЯ ДАНИХ ДЛЯ OLAP-КУБА
# ============================================================================
print("\n1. ГЕНЕРАЦІЯ НАВЧАЛЬНИХ ДАНИХ")
print("-" * 80)

np.random.seed(42)

# Виміри (Dimensions)
years = [2023, 2024, 2025]
quarters = [1, 2, 3, 4]
months = list(range(1, 13))
month_names = ['Січень', 'Лютий', 'Березень', 'Квітень', 'Травень', 'Червень',
               'Липень', 'Серпень', 'Вересень', 'Жовтень', 'Листопад', 'Грудень']

products = ['Ноутбук', 'Смартфон', 'Планшет', 'Навушники', 'Монітор', 'Клавіатура']
categories = {
    'Ноутбук': 'Комп\'ютери',
    'Смартфон': 'Мобільні',
    'Планшет': 'Мобільні',
    'Навушники': 'Аксесуари',
    'Монітор': 'Комп\'ютери',
    'Клавіатура': 'Аксесуари'
}

regions = ['Київ', 'Львів', 'Одеса', 'Харків', 'Дніпро']
countries = {region: 'Україна' for region in regions}

# Генерація таблиці фактів
data = []
for year in years:
    for month in months:
        quarter = (month - 1) // 3 + 1
        for product in products:
            for region in regions:
                # Генерація випадкових продажів з трендом
                base_sales = np.random.randint(20000, 100000)
                trend_factor = 1 + (year - 2023) * 0.1
                seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * month / 12)

                sales = int(base_sales * trend_factor * seasonal_factor)
                quantity = np.random.randint(10, 100)
                profit = int(sales * np.random.uniform(0.15, 0.30))

                data.append({
                    'Дата': pd.Timestamp(year, month, 15),
                    'Рік': year,
                    'Квартал': quarter,
                    'Місяць': month,
                    'Назва_місяця': month_names[month - 1],
                    'Продукт': product,
                    'Категорія': categories[product],
                    'Регіон': region,
                    'Країна': countries[region],
                    'Продажі': sales,
                    'Кількість': quantity,
                    'Прибуток': profit
                })

# Створення DataFrame (це наш OLAP-куб)
df_cube = pd.DataFrame(data)

print("✓ Створено багатовімірний куб даних")
print(f"\n  Загальна кількість записів: {len(df_cube)}")
print(f"  Період: {df_cube['Рік'].min()} - {df_cube['Рік'].max()}")
print(f"  Кількість продуктів: {df_cube['Продукт'].nunique()}")
print(f"  Кількість регіонів: {df_cube['Регіон'].nunique()}")
print(f"\n  Загальні продажі: {df_cube['Продажі'].sum():,} грн")
print(f"  Загальний прибуток: {df_cube['Прибуток'].sum():,} грн")
print(f"  Середній чек: {df_cube['Продажі'].mean():.2f} грн")

print("\n  Структура куба (виміри та факти):")
print(f"    Виміри часу: Рік, Квартал, Місяць")
print(f"    Виміри продукту: Продукт, Категорія")
print(f"    Виміри локації: Регіон, Країна")
print(f"    Факти: Продажі, Кількість, Прибуток")

print("\n  Приклад даних (перші 5 записів):")
print(df_cube.head().to_string(index=False))

# ============================================================================
# 2. ОПЕРАЦІЯ SLICE
# ============================================================================
print("\n" + "=" * 80)
print("2. ОПЕРАЦІЯ SLICE - ВИБІРКА ПО ОДНОМУ ВИМІРУ")
print("=" * 80)

# Slice: вибираємо дані тільки за 2025 рік
slice_year = 2025
df_slice = df_cube[df_cube['Рік'] == slice_year]

print(f"\n✓ Виконано операцію SLICE")
print(f"  Умова: Рік = {slice_year}")
print(f"  Результат: {len(df_slice)} записів (з {len(df_cube)})")
print(f"\n  Продажі за {slice_year} рік: {df_slice['Продажі'].sum():,} грн")
print(f"  Прибуток за {slice_year} рік: {df_slice['Прибуток'].sum():,} грн")

# Аналіз по місяцях для slice
monthly_slice = df_slice.groupby('Назва_місяця').agg({
    'Продажі': 'sum',
    'Прибуток': 'sum'
}).reindex(month_names)

print(f"\n  Розподіл по місяцях {slice_year} року:")
print(monthly_slice.head().to_string())

# ============================================================================
# 3. ОПЕРАЦІЯ DICE
# ============================================================================
print("\n" + "=" * 80)
print("3. ОПЕРАЦІЯ DICE - ВИБІРКА ЗА КІЛЬКОМА КРИТЕРІЯМИ")
print("=" * 80)

# Dice: вибираємо конкретний продукт у певних регіонах за певний рік
dice_conditions = {
    'Рік': 2025,
    'Продукт': 'Ноутбук',
    'Регіон': ['Київ', 'Львів', 'Харків']
}

df_dice = df_cube[
    (df_cube['Рік'] == dice_conditions['Рік']) &
    (df_cube['Продукт'] == dice_conditions['Продукт']) &
    (df_cube['Регіон'].isin(dice_conditions['Регіон']))
    ]

print(f"\n✓ Виконано операцію DICE")
print(f"  Умови:")
print(f"    • Рік = {dice_conditions['Рік']}")
print(f"    • Продукт = {dice_conditions['Продукт']}")
print(f"    • Регіон ∈ {dice_conditions['Регіон']}")
print(f"\n  Результат: {len(df_dice)} записів")
print(f"  Продажі: {df_dice['Продажі'].sum():,} грн")
print(f"  Середня кількість: {df_dice['Кількість'].mean():.1f} шт")

# Деталізація по регіонах
dice_by_region = df_dice.groupby('Регіон').agg({
    'Продажі': 'sum',
    'Кількість': 'sum',
    'Прибуток': 'sum'
}).sort_values('Продажі', ascending=False)

print(f"\n  Деталізація по регіонах:")
print(dice_by_region.to_string())

# ============================================================================
# 4. ОПЕРАЦІЯ ROLL-UP (АГРЕГАЦІЯ)
# ============================================================================
print("\n" + "=" * 80)
print("4. ОПЕРАЦІЯ ROLL-UP - АГРЕГАЦІЯ ДАНИХ")
print("=" * 80)

print("\n4.1. Агрегація з місяців до кварталів")
print("-" * 40)

# Roll-up: з місяців до кварталів
df_rollup_quarter = df_cube.groupby(['Рік', 'Квартал']).agg({
    'Продажі': 'sum',
    'Кількість': 'sum',
    'Прибуток': 'sum'
}).reset_index()

df_rollup_quarter['Період'] = df_rollup_quarter['Рік'].astype(str) + '-Q' + df_rollup_quarter['Квартал'].astype(str)

print("✓ Агреговано дані до рівня кварталів")
print(f"  Записів: {len(df_rollup_quarter)}")
print("\n  Дані по кварталах:")
print(df_rollup_quarter[['Період', 'Продажі', 'Прибуток']].to_string(index=False))

print("\n4.2. Агрегація з кварталів до років")
print("-" * 40)

# Roll-up: з кварталів до років
df_rollup_year = df_cube.groupby('Рік').agg({
    'Продажі': 'sum',
    'Кількість': 'sum',
    'Прибуток': 'sum'
}).reset_index()

print("✓ Агреговано дані до рівня років")
print(f"  Записів: {len(df_rollup_year)}")
print("\n  Дані по роках:")
print(df_rollup_year.to_string(index=False))

# Розрахунок темпів зростання
df_rollup_year['Зростання_продажів_%'] = df_rollup_year['Продажі'].pct_change() * 100
print("\n  Темпи зростання продажів:")
for idx, row in df_rollup_year.iterrows():
    if pd.notna(row['Зростання_продажів_%']):
        print(f"    {row['Рік']}: +{row['Зростання_продажів_%']:.1f}%")

# ============================================================================
# 5. ОПЕРАЦІЯ DRILL-DOWN (ДЕТАЛІЗАЦІЯ)
# ============================================================================
print("\n" + "=" * 80)
print("5. ОПЕРАЦІЯ DRILL-DOWN - ДЕТАЛІЗАЦІЯ ДАНИХ")
print("=" * 80)

print("\n5.1. Деталізація з року до кварталів")
print("-" * 40)

drill_year = 2025
df_drilldown_q = df_cube[df_cube['Рік'] == drill_year].groupby('Квартал').agg({
    'Продажі': 'sum',
    'Прибуток': 'sum'
}).reset_index()

print(f"✓ Деталізовано дані {drill_year} року до кварталів")
print(f"\n  Продажі по кварталах {drill_year} року:")
print(df_drilldown_q.to_string(index=False))

print("\n5.2. Деталізація з кварталу до місяців")
print("-" * 40)

drill_quarter = 2
df_drilldown_m = df_cube[
    (df_cube['Рік'] == drill_year) &
    (df_cube['Квартал'] == drill_quarter)
    ].groupby('Назва_місяця').agg({
    'Продажі': 'sum',
    'Прибуток': 'sum'
}).reset_index()

print(f"✓ Деталізовано Q{drill_quarter} {drill_year} до місяців")
print(f"\n  Продажі по місяцях:")
print(df_drilldown_m.to_string(index=False))

# ============================================================================
# 6. ОПЕРАЦІЯ PIVOT (ПЕРЕСТРУКТУРУВАННЯ)
# ============================================================================
print("\n" + "=" * 80)
print("6. ОПЕРАЦІЯ PIVOT - ПЕРЕСТРУКТУРУВАННЯ ДАНИХ")
print("=" * 80)

print("\n6.1. Pivot: Регіони × Продукти (за 2025 рік)")
print("-" * 40)

# Pivot: регіони по горизонталі, продукти по вертикалі
pivot_region_product = df_cube[df_cube['Рік'] == 2025].pivot_table(
    values='Продажі',
    index='Продукт',
    columns='Регіон',
    aggfunc='sum',
    fill_value=0
)

print("✓ Створено pivot-таблицю")
print("\n  Продажі по регіонах та продуктах (грн):")
print(pivot_region_product.to_string())

print("\n  Топ-3 регіони по загальних продажах:")
region_totals = pivot_region_product.sum(axis=0).sort_values(ascending=False)
for i, (region, sales) in enumerate(region_totals.head(3).items(), 1):
    print(f"    {i}. {region}: {sales:,} грн")

print("\n6.2. Pivot: Категорії × Квартали (всі роки)")
print("-" * 40)

pivot_category_quarter = df_cube.pivot_table(
    values='Продажі',
    index='Категорія',
    columns=['Рік', 'Квартал'],
    aggfunc='sum',
    fill_value=0
)

print("✓ Створено pivot-таблицю категорій по кварталах")
print(f"\n  Розмір таблиці: {pivot_category_quarter.shape}")
print("\n  Перші стовпці:")
print(pivot_category_quarter.iloc[:, :6].to_string())

# ============================================================================
# 7. ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ
# ============================================================================
print("\n" + "=" * 80)
print("7. ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ OLAP-ОПЕРАЦІЙ")
print("=" * 80)

fig = plt.figure(figsize=(20, 12))

# 7.1. Візуалізація SLICE - продажі по місяцях 2025
ax1 = plt.subplot(3, 3, 1)
monthly_sales = df_slice.groupby('Місяць')['Продажі'].sum()
ax1.plot(monthly_sales.index, monthly_sales.values, marker='o', linewidth=2, markersize=8)
ax1.set_title(f'SLICE: Продажі по місяцях {slice_year} року', fontsize=12, fontweight='bold')
ax1.set_xlabel('Місяць')
ax1.set_ylabel('Продажі (грн)')
ax1.grid(True, alpha=0.3)

# 7.2. Візуалізація DICE - продажі по регіонах
ax2 = plt.subplot(3, 3, 2)
dice_region_sales = df_dice.groupby('Регіон')['Продажі'].sum().sort_values()
dice_region_sales.plot(kind='barh', ax=ax2, color='coral')
ax2.set_title('DICE: Продажі ноутбуків по регіонах', fontsize=12, fontweight='bold')
ax2.set_xlabel('Продажі (грн)')
for i, v in enumerate(dice_region_sales.values):
    ax2.text(v, i, f' {v:,.0f}', va='center')

# 7.3. Візуалізація ROLL-UP - продажі по роках
ax3 = plt.subplot(3, 3, 3)
colors = ['#3498db', '#2ecc71', '#e74c3c']
bars = ax3.bar(df_rollup_year['Рік'], df_rollup_year['Продажі'], color=colors, alpha=0.8)
ax3.set_title('ROLL-UP: Агреговані продажі по роках', fontsize=12, fontweight='bold')
ax3.set_xlabel('Рік')
ax3.set_ylabel('Продажі (грн)')
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:,.0f}', ha='center', va='bottom')

# 7.4. Візуалізація DRILL-DOWN - квартали
ax4 = plt.subplot(3, 3, 4)
df_drilldown_q.plot(x='Квартал', y=['Продажі', 'Прибуток'],
                    kind='bar', ax=ax4, rot=0)
ax4.set_title(f'DRILL-DOWN: Квартали {drill_year} року', fontsize=12, fontweight='bold')
ax4.set_xlabel('Квартал')
ax4.set_ylabel('Сума (грн)')
ax4.legend(['Продажі', 'Прибуток'])

# 7.5. Heatmap продажів по продуктах та регіонах
ax5 = plt.subplot(3, 3, 5)
sns.heatmap(pivot_region_product, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax5, cbar_kws={'label': 'Продажі (грн)'})
ax5.set_title('PIVOT: Heatmap продажів (Продукти × Регіони)', fontsize=12, fontweight='bold')

# 7.6. Продажі по категоріях (pie chart)
ax6 = plt.subplot(3, 3, 6)
category_sales = df_cube.groupby('Категорія')['Продажі'].sum()
colors_pie = ['#ff9999', '#66b3ff', '#99ff99']
wedges, texts, autotexts = ax6.pie(category_sales, labels=category_sales.index,
                                   autopct='%1.1f%%', colors=colors_pie, startangle=90)
ax6.set_title('Розподіл продажів по категоріях', fontsize=12, fontweight='bold')

# 7.7. Динаміка продажів по кварталах (всі роки)
ax7 = plt.subplot(3, 3, 7)
for year in years:
    year_data = df_rollup_quarter[df_rollup_quarter['Рік'] == year]
    ax7.plot(year_data['Квартал'], year_data['Продажі'], marker='o', label=f'{year}', linewidth=2)
ax7.set_title('ROLL-UP: Динаміка по кварталах', fontsize=12, fontweight='bold')
ax7.set_xlabel('Квартал')
ax7.set_ylabel('Продажі (грн)')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 7.8. Топ-5 продуктів
ax8 = plt.subplot(3, 3, 8)
top_products = df_cube.groupby('Продукт')['Продажі'].sum().sort_values(ascending=True).tail(5)
top_products.plot(kind='barh', ax=ax8, color='lightgreen')
ax8.set_title('Топ-5 продуктів за продажами', fontsize=12, fontweight='bold')
ax8.set_xlabel('Продажі (грн)')

# 7.9. Порівняння прибутковості по регіонах
ax9 = plt.subplot(3, 3, 9)
region_profit_margin = df_cube.groupby('Регіон').agg({
    'Продажі': 'sum',
    'Прибуток': 'sum'
})
region_profit_margin['Маржа_%'] = (region_profit_margin['Прибуток'] / region_profit_margin['Продажі']) * 100
region_profit_margin['Маржа_%'].sort_values(ascending=True).plot(kind='barh', ax=ax9, color='purple', alpha=0.7)
ax9.set_title('Прибутковість по регіонах (%)', fontsize=12, fontweight='bold')
ax9.set_xlabel('Маржа (%)')

plt.tight_layout()
plt.savefig('lab16_olap_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Графіки збережено у файл 'lab16_olap_results.png'")

# ============================================================================
# 8. АНАЛІТИЧНІ ЗВІТИ
# ============================================================================
print("\n" + "=" * 80)
print("8. АНАЛІТИЧНІ ЗВІТИ")
print("=" * 80)

print("\n8.1. Звіт: Продажі по роках та категоріях")
print("-" * 40)

report_year_category = df_cube.pivot_table(
    values='Продажі',
    index='Категорія',
    columns='Рік',
    aggfunc='sum',
    margins=True,
    margins_name='ВСЬОГО'
)

print(report_year_category.to_string())

print("\n8.2. Звіт: Топ-10 комбінацій (Регіон + Продукт)")
print("-" * 40)

top_combinations = df_cube.groupby(['Регіон', 'Продукт']).agg({
    'Продажі': 'sum',
    'Кількість': 'sum'
}).sort_values('Продажі', ascending=False).head(10)

print(top_combinations.to_string())

print("\n8.3. Звіт: Ефективність по регіонах")
print("-" * 40)

region_efficiency = df_cube.groupby('Регіон').agg({
    'Продажі': 'sum',
    'Прибуток': 'sum',
    'Кількість': 'sum'
})
region_efficiency['Середній_чек'] = region_efficiency['Продажі'] / region_efficiency['Кількість']
region_efficiency['Маржа_%'] = (region_efficiency['Прибуток'] / region_efficiency['Продажі']) * 100
region_efficiency = region_efficiency.sort_values('Продажі', ascending=False)

print(region_efficiency.to_string())

# ============================================================================
# 9. ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ
# ============================================================================
print("\n" + "=" * 80)
print("9. ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ")
print("=" * 80)

# Збереження куба даних
df_cube.to_csv('olap_cube_data.csv', index=False, encoding='utf-8-sig')
print("✓ Куб даних збережено: 'olap_cube_data.csv'")

# Збереження результатів операцій
df_slice.to_csv('olap_slice_result.csv', index=False, encoding='utf-8-sig')
print("✓ Результат SLICE збережено: 'olap_slice_result.csv'")

df_dice.to_csv('olap_dice_result.csv', index=False, encoding='utf-8-sig')
print("✓ Результат DICE збережено: 'olap_dice_result.csv'")

df_rollup_year.to_csv('olap_rollup_result.csv', index=False, encoding='utf-8-sig')
print("✓ Результат ROLL-UP збережено: 'olap_rollup_result.csv'")

pivot_region_product.to_csv('olap_pivot_result.csv', encoding='utf-8-sig')
print("✓ Результат PIVOT збережено: 'olap_pivot_result.csv'")

plt.show()
