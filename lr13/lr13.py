import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from urllib.parse import urljoin, urlparse
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. НАЛАШТУВАННЯ ТА ПІДГОТОВКА
# ============================================================================
print("\n1. НАЛАШТУВАННЯ ТА ПІДГОТОВКА")
print("-" * 80)

# Заголовки для імітації браузера
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
}


def check_robots_txt(url):
    """Перевірка robots.txt"""
    parsed_url = urlparse(url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"

    try:
        response = requests.get(robots_url, timeout=5)
        if response.status_code == 200:
            print(f"✓ robots.txt знайдено: {robots_url}")
            print(f"  Перші 500 символів:")
            print(f"  {response.text[:500]}...")
            return True
        else:
            print(f"✗ robots.txt не знайдено")
            return False
    except Exception as e:
        print(f"✗ Помилка при перевірці robots.txt: {e}")
        return False


def get_page_content(url, timeout=10):
    """Отримання HTML-коду сторінки"""
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        return response.text, response.status_code
    except requests.exceptions.RequestException as e:
        print(f"✗ Помилка при отриманні сторінки: {e}")
        return None, None


# ============================================================================
# 2. ПАРСИНГ ОСНОВНОЇ СТОРІНКИ (Google.com)
# ============================================================================
print("\n2. ПАРСИНГ GOOGLE.COM")
print("-" * 80)

google_url = "https://www.google.com"
print(f"URL: {google_url}")

# Перевірка robots.txt
check_robots_txt(google_url)

# Отримання сторінки
html_content, status_code = get_page_content(google_url)

if html_content:
    print(f"\n✓ Сторінка успішно завантажена")
    print(f"  Статус код: {status_code}")
    print(f"  Розмір: {len(html_content)} байт")

    # Парсинг за допомогою BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Базова інформація про сторінку
    google_data = {
        'URL': google_url,
        'Заголовок': soup.title.string if soup.title else 'Не знайдено',
        'Мова': soup.html.get('lang', 'Не вказано') if soup.html else 'Не вказано',
        'Кількість посилань': len(soup.find_all('a')),
        'Кількість зображень': len(soup.find_all('img')),
        'Кількість скриптів': len(soup.find_all('script')),
        'Кількість стилів': len(soup.find_all('link', rel='stylesheet')),
    }

    print("\nІнформація про сторінку Google:")
    for key, value in google_data.items():
        print(f"  {key}: {value}")

    # Витягування meta-тегів
    meta_tags = soup.find_all('meta')
    print(f"\n✓ Знайдено {len(meta_tags)} meta-тегів")

    meta_data = []
    for meta in meta_tags[:10]:  # Перші 10
        meta_info = {
            'name': meta.get('name', meta.get('property', 'N/A')),
            'content': meta.get('content', 'N/A')[:100]  # Обмежуємо довжину
        }
        meta_data.append(meta_info)

    # Витягування посилань
    links = soup.find_all('a', href=True)
    print(f"\n✓ Знайдено {len(links)} посилань")

    links_data = []
    for link in links[:20]:  # Перші 20
        href = link.get('href', '')
        text = link.get_text(strip=True)[:50]
        if href and not href.startswith('#'):
            links_data.append({
                'Текст': text if text else '[Без тексту]',
                'URL': href[:100],
                'Повний URL': urljoin(google_url, href)[:100]
            })

# ============================================================================
# 3. ПАРСИНГ ДОДАТКОВОГО САЙТУ (Приклад з новинами)
# ============================================================================
print("\n3. ПАРСИНГ ДОДАТКОВОГО САЙТУ (ДЕМОНСТРАЦІЯ)")
print("-" * 80)

# Використовуємо Wikipedia як приклад (більш дружній до парсингу)
demo_url = "https://en.wikipedia.org/wiki/Web_scraping"
print(f"URL: {demo_url}")

html_demo, status_demo = get_page_content(demo_url)

demo_articles = []
if html_demo:
    soup_demo = BeautifulSoup(html_demo, 'html.parser')

    print(f"✓ Сторінка завантажена")
    print(f"  Заголовок: {soup_demo.title.string}")

    # Парсинг заголовків
    headings = soup_demo.find_all(['h1', 'h2', 'h3'])
    print(f"\n✓ Знайдено {len(headings)} заголовків")

    for i, heading in enumerate(headings[:15], 1):
        text = heading.get_text(strip=True)
        if text and len(text) > 3:
            demo_articles.append({
                'Номер': i,
                'Тип': heading.name,
                'Заголовок': text,
                'Довжина': len(text)
            })

    # Парсинг параграфів
    paragraphs = soup_demo.find_all('p')
    print(f"✓ Знайдено {len(paragraphs)} параграфів")

    # Парсинг зовнішніх посилань
    external_links = []
    for link in soup_demo.find_all('a', href=True):
        href = link.get('href')
        if href.startswith('http') and 'wikipedia.org' not in href:
            external_links.append({
                'Текст': link.get_text(strip=True)[:50],
                'URL': href[:100]
            })

    print(f"✓ Знайдено {len(external_links)} зовнішніх посилань")

# ============================================================================
# 4. СТВОРЕННЯ СИНТЕТИЧНИХ ДАНИХ ДЛЯ АНАЛІЗУ
# ============================================================================
print("\n4. ПІДГОТОВКА ДАНИХ ДЛЯ АНАЛІЗУ")
print("-" * 80)

# Створюємо структуровані дані для аналізу
all_data = {
    'Статистика сторінок': pd.DataFrame([
        {'Сайт': 'Google.com', 'Посилання': google_data['Кількість посилань'],
         'Зображення': google_data['Кількість зображень'],
         'Скрипти': google_data['Кількість скриптів']},
        {'Сайт': 'Wikipedia (demo)', 'Посилання': len(external_links) if html_demo else 0,
         'Зображення': len(soup_demo.find_all('img')) if html_demo else 0,
         'Скрипти': len(soup_demo.find_all('script')) if html_demo else 0}
    ])
}

# Дані про заголовки з demo сайту
if demo_articles:
    headings_df = pd.DataFrame(demo_articles)
    print(f"\n✓ Створено таблицю з {len(headings_df)} заголовками")

# Дані про посилання з Google
if links_data:
    links_df = pd.DataFrame(links_data)
    print(f"✓ Створено таблицю з {len(links_df)} посиланнями")

# Дані про meta-теги
if meta_data:
    meta_df = pd.DataFrame(meta_data)
    print(f"✓ Створено таблицю з {len(meta_df)} meta-тегами")

# ============================================================================
# 5. ЗБЕРЕЖЕННЯ ДАНИХ
# ============================================================================
print("\n5. ЗБЕРЕЖЕННЯ ДАНИХ")
print("-" * 80)

# Збереження у CSV
try:
    if links_data:
        links_df.to_csv('google_links.csv', index=False, encoding='utf-8-sig')
        print("✓ google_links.csv збережено")

    if demo_articles:
        headings_df.to_csv('demo_headings.csv', index=False, encoding='utf-8-sig')
        print("✓ demo_headings.csv збережено")

    if meta_data:
        meta_df.to_csv('google_meta_tags.csv', index=False, encoding='utf-8-sig')
        print("✓ google_meta_tags.csv збережено")

    all_data['Статистика сторінок'].to_csv('pages_statistics.csv', index=False, encoding='utf-8-sig')
    print("✓ pages_statistics.csv збережено")
except Exception as e:
    print(f"✗ Помилка при збереженні CSV: {e}")

# Збереження у JSON
try:
    json_data = {
        'google_info': google_data,
        'meta_tags': meta_data[:5],
        'links_sample': links_data[:5],
        'demo_headings': demo_articles[:5]
    }

    with open('web_scraping_data.json', 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print("✓ web_scraping_data.json збережено")
except Exception as e:
    print(f"✗ Помилка при збереженні JSON: {e}")

# ============================================================================
# 6. АНАЛІЗ ТА ВІЗУАЛІЗАЦІЯ
# ============================================================================
print("\n6. АНАЛІЗ ТА ВІЗУАЛІЗАЦІЯ")
print("-" * 80)

fig = plt.figure(figsize=(16, 10))

# 6.1. Статистика елементів на сторінках
ax1 = plt.subplot(2, 3, 1)
stats_df = all_data['Статистика сторінок']
stats_df.set_index('Сайт')[['Посилання', 'Зображення', 'Скрипти']].plot(
    kind='bar', ax=ax1, width=0.7, alpha=0.8
)
plt.title('Статистика елементів на сторінках', fontsize=12, fontweight='bold')
plt.ylabel('Кількість')
plt.xlabel('Сайт')
plt.xticks(rotation=45, ha='right')
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=0.3)

# 6.2. Розподіл типів заголовків (якщо є)
if demo_articles:
    ax2 = plt.subplot(2, 3, 2)
    heading_counts = headings_df['Тип'].value_counts()
    colors = plt.cm.Set3(range(len(heading_counts)))
    plt.pie(heading_counts.values, labels=heading_counts.index,
            autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Розподіл типів заголовків\n(Demo сайт)', fontsize=12, fontweight='bold')

# 6.3. Довжина заголовків
if demo_articles:
    ax3 = plt.subplot(2, 3, 3)
    plt.hist(headings_df['Довжина'], bins=15, color='skyblue',
             edgecolor='black', alpha=0.7)
    plt.axvline(headings_df['Довжина'].mean(), color='red',
                linestyle='--', linewidth=2, label=f"Середнє: {headings_df['Довжина'].mean():.1f}")
    plt.title('Розподіл довжини заголовків', fontsize=12, fontweight='bold')
    plt.xlabel('Довжина (символи)')
    plt.ylabel('Частота')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

# 6.4. Таблиця з прикладами посилань
ax4 = plt.subplot(2, 3, 4)
ax4.axis('tight')
ax4.axis('off')
if links_data:
    table_data = [[row['Текст'][:30], row['URL'][:40]]
                  for i, row in enumerate(links_data[:8])]
    table = ax4.table(cellText=table_data,
                      colLabels=['Текст посилання', 'URL'],
                      cellLoc='left', loc='center',
                      colWidths=[0.4, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    plt.title('Приклади посилань з Google', fontsize=12, fontweight='bold', pad=20)

# 6.5. Порівняльна діаграма
ax5 = plt.subplot(2, 3, 5)
categories = ['Посилання', 'Зображення', 'Скрипти']
google_values = [google_data['Кількість посилань'],
                 google_data['Кількість зображень'],
                 google_data['Кількість скриптів']]

x = range(len(categories))
bars = plt.bar(x, google_values, color=['#4285F4', '#EA4335', '#FBBC04'], alpha=0.8)
plt.title('Елементи на сторінці Google.com', fontsize=12, fontweight='bold')
plt.xlabel('Тип елементу')
plt.ylabel('Кількість')
plt.xticks(x, categories)
plt.grid(axis='y', alpha=0.3)

# Додаємо значення на стовпчики
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 6.6. Інформаційна панель
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

info_text = f"""
ПІДСУМКОВА ІНФОРМАЦІЯ

Google.com:
• Заголовок: {google_data['Заголовок'][:40]}
• Мова: {google_data['Мова']}
• Посилання: {google_data['Кількість посилань']}
• Зображення: {google_data['Кількість зображень']}
• Meta-теги: {len(meta_data)}

Demo сайт (Wikipedia):
• Заголовків: {len(demo_articles) if demo_articles else 0}
• Параграфів: {len(paragraphs) if html_demo else 0}
• Зовнішніх посилань: {len(external_links) if html_demo else 0}

Файли збережено:
✓ google_links.csv
✓ demo_headings.csv
✓ google_meta_tags.csv
✓ pages_statistics.csv
✓ web_scraping_data.json
"""

plt.text(0.1, 0.95, info_text, transform=ax6.transAxes,
         fontsize=10, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('lab13_web_scraping_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Графіки збережено у файл 'lab13_web_scraping_results.png'")

# Виведення прикладу даних
if links_data:
    print("\nПРИКЛАД ЗІБРАНИХ ДАНИХ (перші 5 посилань):")
    print(links_df.head().to_string(index=False))

if demo_articles:
    print("\nПРИКЛАД ЗАГОЛОВКІВ (перші 5):")
    print(headings_df.head().to_string(index=False))