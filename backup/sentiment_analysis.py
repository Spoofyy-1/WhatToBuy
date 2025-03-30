import requests

ALPHA_VANTAGE_API_KEY = 'C7QTPU86YRN5EVEY'

def fetch_news_from_alpha_vantage(symbol, date=None):
    """
    Get news and sentiment data from Alpha Vantage's News & Sentiment API
    
    :param symbol: Stock symbol to fetch news for.
    :param date: Date to fetch news for (optional).
    :return: List of news articles with sentiment data.
    """
    try:
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        
        response = requests.get(url)
        data = response.json()
        
        if "feed" not in data:
            print(f"Error: {data.get('Error Message', 'No news data available')}")
            return []
            
        articles = []
        for item in data["feed"][:10]:  # Limit to 10 articles
            # Safe extraction of ticker sentiment
            ticker_sentiment_data = {"ticker_sentiment_score": 0, "ticker_sentiment_label": "neutral"}
            for ticker_data in item.get("ticker_sentiment", []):
                if ticker_data.get("ticker") == symbol:
                    try:
                        ticker_sentiment_data = {
                            "ticker_sentiment_score": float(ticker_data.get("ticker_sentiment_score", 0)),
                            "ticker_sentiment_label": ticker_data.get("ticker_sentiment_label", "neutral")
                        }
                    except (ValueError, TypeError):
                        # In case of errors, use default neutral values
                        ticker_sentiment_data = {"ticker_sentiment_score": 0, "ticker_sentiment_label": "neutral"}
                    break
            
            try:
                overall_sentiment = float(item.get("overall_sentiment_score", 0))
            except (ValueError, TypeError):
                overall_sentiment = 0
                
            article = {
                "title": item.get("title", ""),
                "source": {"name": item.get("source", "Alpha Vantage")},
                "publishedAt": item.get("time_published", ""),
                "url": item.get("url", ""),
                "description": item.get("summary", ""),
                "sentiment_score": overall_sentiment,
                "sentiment_label": item.get("overall_sentiment_label", "neutral"),
                "ticker_sentiment": ticker_sentiment_data
            }
            articles.append(article)
            
        return articles
        
    except Exception as e:
        print(f"Error fetching news from Alpha Vantage: {e}")
        return []

def analyze_sentiment(articles):
    """
    Analyze sentiment of news articles.
    
    :param articles: List of news articles.
    :return: Average sentiment score.
    """
    if not articles:
        return 0
    
    sentiment_scores = []
    for article in articles:
        # Get sentiment score from ticker_sentiment
        ticker_sentiment = article.get('ticker_sentiment', {})
        try:
            sentiment_score = float(ticker_sentiment.get('ticker_sentiment_score', 0))
            sentiment_scores.append(sentiment_score)
        except (ValueError, TypeError):
            # Skip invalid scores
            continue
    
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0 