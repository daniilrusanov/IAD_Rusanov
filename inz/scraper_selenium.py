from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

def get_rozetka_reviews_selenium(product_url, max_clicks=10):
    """Scrapes Rozetka reviews using Selenium with explicit waits."""
    
    if 'comments' not in product_url:
        product_url = product_url.rstrip('/') + '/comments/'
    
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    
    driver = None
    reviews_data = []
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        wait = WebDriverWait(driver, 10)
        
        print(f"Завантаження {product_url}...")
        driver.get(product_url)
        
        # Wait for reviews to load
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'li.product-comments__list-item')))
            print("Відгуки завантажено")
        except:
            print("Не вдалося знайти відгуки")
            return pd.DataFrame()
        
        time.sleep(2)
        
        clicks = 0
        previous_count = 0
        
        while clicks < max_clicks:
            current_count = len(driver.find_elements(By.CSS_SELECTOR, 'li.product-comments__list-item'))
            
            if current_count == previous_count and clicks > 0:
                print(f"Кількість не змінилась: {current_count}")
                break
            
            previous_count = current_count
            
            # Find "Показати ще" button
            show_more = None
            try:
                # Try by class first (most reliable)
                try:
                    show_more = driver.find_element(By.CSS_SELECTOR, 'button.toggle-btn-down')
                    if not show_more.is_displayed():
                        show_more = None
                except:
                    pass
                
                # Fallback: search by text
                if not show_more:
                    buttons = driver.find_elements(By.TAG_NAME, 'button')
                    for btn in buttons:
                        try:
                            text = btn.text.strip()
                            if text == "Показати ще" and btn.is_displayed():
                                show_more = btn
                                break
                        except:
                            continue
            except:
                pass
            
            if show_more:
                try:
                    # Scroll to button
                    driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", show_more)
                    time.sleep(0.5)
                    
                    # Click
                    driver.execute_script("arguments[0].click();", show_more)
                    clicks += 1
                    print(f"Клік {clicks}: {current_count} відгуків")
                    
                    # Wait for new reviews to load
                    time.sleep(2)
                except Exception as e:
                    print(f"Помилка кліку: {e}")
                    break
            else:
                print(f"Кнопка не знайдена. Всього: {current_count}")
                break
        
        # Parse all reviews
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        items = soup.find_all('li', class_='product-comments__list-item')
        print(f"Парсинг {len(items)} відгуків...")
        
        for item in items:
            try:
                # Stars
                stars = 0
                stars_div = item.find('div', class_='stars__rating')
                if stars_div and stars_div.get('style'):
                    match = re.search(r'(\d+)%', stars_div.get('style'))
                    if match:
                        stars = round(int(match.group(1)) / 20)
                
                # Date
                date_elem = item.find('p', class_='date')
                date_text = date_elem.get_text(strip=True) if date_elem else ""
                
                # Text
                text_elem = item.find('div', class_='comment__body')
                review_text = text_elem.get_text(" ", strip=True) if text_elem else ""
                
                if review_text:
                    reviews_data.append({
                        'text': review_text,
                        'stars': stars,
                        'date': date_text
                    })
            except:
                continue
        
    except Exception as e:
        print(f"Помилка: {e}")
    finally:
        if driver:
            driver.quit()
    
    df = pd.DataFrame(reviews_data)
    if not df.empty:
        df = df.drop_duplicates(subset=['text', 'date'])
    
    print(f"Зібрано {len(df)} унікальних відгуків")
    return df

if __name__ == "__main__":
    url = "https://bt.rozetka.com.ua/ua/bosch-twk1m121/p412020471/comments/"
    df = get_rozetka_reviews_selenium(url, max_clicks=10)
    print(f"\nРезультат: {len(df)} відгуків")
    if not df.empty:
        print(df['stars'].value_counts())
