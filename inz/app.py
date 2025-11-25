from flask import Flask, render_template, request
from scraper_selenium import get_rozetka_reviews_selenium
from analysis import analyze_sentiment, create_plots, get_keywords

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form.get('url')
        
        if not url:
             return render_template('index.html', error="Будь ласка, введіть URL")

        # 1. Scraping with Selenium
        try:
            reviews_df = get_rozetka_reviews_selenium(url, max_clicks=10)
        except Exception as e:
            return render_template('index.html', error=f"Помилка при скрапінгу: {str(e)}")
        
        if reviews_df.empty:
            return render_template('index.html', error="Відгуки не знайдено або невірне посилання. Переконайтеся, що це сторінка товару Rozetka.")

        # 2. Analysis
        stats, sentiment_df = analyze_sentiment(reviews_df)
        
        # Keywords
        pos_keywords = get_keywords(sentiment_df[sentiment_df['sentiment'] == 'Позитивний']['text'])
        neg_keywords = get_keywords(sentiment_df[sentiment_df['sentiment'] == 'Негативний']['text'])
        
        # 3. Visualization
        plots = create_plots(sentiment_df)
        
        return render_template('index.html', 
                               results=True, 
                               stats=stats, 
                               plots=plots,
                               url=url,
                               pos_keywords=pos_keywords,
                               neg_keywords=neg_keywords)
                               
    return render_template('index.html', results=False)

if __name__ == '__main__':
    app.run(debug=True)
