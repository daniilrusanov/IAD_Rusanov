import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
matplotlib.use('Agg') # Fix for Flask threading issue
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import io
import base64
import pymorphy2
import os

def load_stopwords():
    """
    Load Ukrainian stopwords from CSV file.
    Returns a list of stopwords.
    """
    stopwords_path = os.path.join(os.path.dirname(__file__), 'stopwords_ua.csv')
    try:
        import csv
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return [row['word'] for row in reader]
    except FileNotFoundError:
        print(f"Warning: stopwords file not found at {stopwords_path}, using minimal stopwords")
        return ['і', 'в', 'на', 'з', 'що', 'не', 'до', 'у', 'я', 'це', 'за', 'та', 'а']
    except Exception as e:
        print(f"Error loading stopwords: {e}")
        return []

def clean_text(text):
    """
    Cleans text: lowercase, remove punctuation, remove extra spaces.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_sentiment(df):
    """
    Analyzes sentiment based on stars and prepares stats.
    
    Args:
        df (pd.DataFrame): DataFrame with 'stars' and 'text'.
        
    Returns:
        tuple: (stats dict, enriched DataFrame)
    """
    if df.empty:
        return {}, df
        
    # Sentiment Classification based on stars (Ukrainian labels)
    def classify(stars):
        if stars >= 4:
            return 'Позитивний'
        elif stars == 3:
            return 'Нейтральний'
        else:
            return 'Негативний'
            
    df['sentiment'] = df['stars'].apply(classify)
    
    # Basic Stats
    # Exclude reviews with 0 stars (failed extraction) from average calculation
    valid_ratings = df[df['stars'] > 0]
    
    stats = {
        'total_reviews': len(df),
        'average_rating': round(valid_ratings['stars'].mean(), 2) if not valid_ratings.empty else 0.0,
        'positive_count': len(df[df['sentiment'] == 'Позитивний']),
        'neutral_count': len(df[df['sentiment'] == 'Нейтральний']),
        'negative_count': len(df[df['sentiment'] == 'Негативний'])
    }
    
    return stats, df

def get_keywords(text_series, top_n=10):
    """
    Extracts top keywords using TF-IDF.
    """
    if text_series.empty:
        return []
        
    # Load stopwords from CSV
    stop_words = load_stopwords()
    
    try:
        tfidf = TfidfVectorizer(stop_words=stop_words, max_features=top_n)
        tfidf_matrix = tfidf.fit_transform(text_series)
        feature_names = tfidf.get_feature_names_out()
        
        # Sum tfidf scores for each term
        sums = tfidf_matrix.sum(axis=0)
        data = []
        for col, term in enumerate(feature_names):
            data.append( (term, sums[0, col] ))
            
        ranking = pd.DataFrame(data, columns=['term', 'rank'])
        ranking = ranking.sort_values('rank', ascending=False)
        return ranking['term'].head(top_n).tolist()
    except ValueError:
        # Handle case with empty vocabulary or too few documents
        return []

def create_plots(df):
    """
    Generates plots and returns them as base64 strings.
    """
    plots = {}
    
    if df.empty:
        return plots
        
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Pie Chart (Sentiment) - Ukrainian labels
    plt.figure(figsize=(6, 6))
    sentiment_counts = df['sentiment'].value_counts()
    colors = {'Позитивний': '#66b3ff', 'Нейтральний': '#99ff99', 'Негативний': '#ff9999'}
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', 
            colors=[colors.get(x, '#cccccc') for x in sentiment_counts.index], startangle=90)
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plots['pie_chart'] = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # 2. Histogram (Stars) - Modern colors
    plt.figure(figsize=(8, 5))
    valid_stars = df[df['stars'] > 0]['stars']  # Exclude 0 stars
    star_counts = valid_stars.value_counts().sort_index()
    # Modern gradient palette
    colors_bars = ['#ee6666', '#fac858', '#73c0de', '#5470c6', '#91cc75']
    plt.bar(star_counts.index, star_counts.values, color=colors_bars[:len(star_counts)], edgecolor='white', linewidth=2, alpha=0.9)
    plt.xlabel('Зірки')
    plt.ylabel('Кількість')
    plt.xticks([1, 2, 3, 4, 5])
    plt.grid(axis='y', alpha=0.3)
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plots['histogram'] = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # 3. Purchase Recommendation
    plt.figure(figsize=(8, 5))
    
    # Calculate recommendation score (0-100)
    mean_rating = df[df['stars'] > 0]['stars'].mean()
    positive_ratio = len(df[df['sentiment'] == 'Позитивний']) / len(df) * 100
    
    # Weighted score: 60% from rating, 40% from sentiment
    recommendation_score = (mean_rating / 5 * 60) + (positive_ratio * 0.4)
    
    # Determine recommendation level - softer modern colors
    if recommendation_score >= 80:
        recommendation = 'Дуже рекомендуємо'
        color = '#5470c6'  # Soft blue
    elif recommendation_score >= 60:
        recommendation = 'Рекомендуємо'
        color = '#91cc75'  # Soft green
    elif recommendation_score >= 40:
        recommendation = 'Нейтрально'
        color = '#fac858'  # Soft yellow
    else:
        recommendation = 'Не рекомендуємо'
        color = '#ee6666'  # Soft red
    
    # Create gauge-style chart
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Background segments - lighter pastel colors
    segments = [20, 20, 20, 40]
    segment_colors = ['#ffcdd2', '#fff9c4', '#c8e6c9', '#bbdefb']  # Light pastel
    
    # Draw background
    bottom = 0
    for seg, seg_color in zip(segments, segment_colors):
        ax.barh(0, seg, left=bottom, height=0.5, color=seg_color, edgecolor='#e0e0e0', linewidth=1)
        bottom += seg
    
    # Draw indicator
    ax.barh(0, recommendation_score, height=0.5, color=color, edgecolor='#424242', linewidth=2, alpha=0.9)
    
    # Add score text - dark text for better visibility
    ax.text(recommendation_score/2, 0, f'{recommendation_score:.1f}%', 
            ha='center', va='center', fontsize=18, fontweight='bold', color='#212121')
    
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Рівень рекомендації (%)', fontsize=12)
    ax.set_title(f'Рекомендація: {recommendation}', fontsize=16, fontweight='bold')
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plots['sentiment_by_stars'] = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # 4. Review Length Distribution (Line Plot)
    plt.figure(figsize=(8, 5))
    df['text_length'] = df['text'].str.len()
    
    # Create histogram data
    counts, bins = np.histogram(df['text_length'], bins=30)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Plot as line with gradient fill
    plt.plot(bin_centers, counts, color='#5470c6', linewidth=2.5, marker='o', markersize=4)
    plt.fill_between(bin_centers, counts, alpha=0.4, color='#5470c6')
    plt.xlabel('Кількість символів')
    plt.ylabel('Кількість відгуків')
    plt.grid(True, alpha=0.3)
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plots['review_length'] = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # 5. Word Clouds
    # Clean text for word clouds
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Load stopwords from CSV
    stop_words = load_stopwords()

    def generate_wordcloud(text_data, title, color_func=None):
        if not text_data:
            return None
        text = " ".join(text_data)
        if not text.strip():
            return None
            
        wc = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words, colormap=color_func).generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()
        
    # Positive Word Cloud
    pos_text = df[df['sentiment'] == 'Позитивний']['cleaned_text'].tolist()
    plots['wordcloud_pos'] = generate_wordcloud(pos_text, 'Хмара слів позитивних відгуків', 'Greens')
    
    # Negative Word Cloud
    neg_text = df[df['sentiment'] == 'Негативний']['cleaned_text'].tolist()
    plots['wordcloud_neg'] = generate_wordcloud(neg_text, 'Хмара слів негативних відгуків', 'Reds')
    
    plt.close('all')
    
    return plots
