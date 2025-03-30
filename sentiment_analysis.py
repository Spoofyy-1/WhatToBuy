import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import random
import time
import traceback

ALPHA_VANTAGE_API_KEY = 'C7QTPU86YRN5EVEY'
SENTIMENT_CACHE_FILE = 'sentiment_cache.json'

# Global cache for sentiment data
news_cache = {}

def get_sentiment_cache():
    """
    Get the sentiment cache from disk or create a new one.
    
    Returns:
        dict: Sentiment cache
    """
    cache = {}
    if os.path.exists(SENTIMENT_CACHE_FILE):
        try:
            with open(SENTIMENT_CACHE_FILE, 'r') as f:
                cache = json.load(f)
            print(f"Loaded sentiment cache with {len(cache)} symbols")
        except Exception as e:
            print(f"Error loading sentiment cache: {e}")
            cache = {}
    
    return cache

def save_sentiment_cache(cache):
    """
    Save the sentiment cache to disk.
    
    Args:
        cache (dict): Sentiment cache
    """
    try:
        with open(SENTIMENT_CACHE_FILE, 'w') as f:
            json.dump(cache, f)
        print(f"Saved sentiment cache with {len(cache)} symbols")
    except Exception as e:
        print(f"Error saving sentiment cache: {e}")

def fetch_news_from_alpha_vantage(symbol, date=None):
    """
    Get news and sentiment data from Alpha Vantage's News & Sentiment API
    
    :param symbol: Stock symbol to fetch news for.
    :param date: Date to fetch news for (optional).
    :return: List of news articles with sentiment data.
    """
    try:
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol.upper()}&apikey={ALPHA_VANTAGE_API_KEY}"
        
        # If date is specified, limit to that date (using a 24-hour window)
        if date:
            # Convert date to pandas Timestamp if it's not already
            if not isinstance(date, pd.Timestamp):
                date = pd.Timestamp(date)
            
            # Format for Alpha Vantage: YYYYMMDDTHHMM
            time_from = date.strftime('%Y%m%dT0000')
            
            # Set time_to to 23:59 on the same day
            time_to = date.strftime('%Y%m%dT2359')
            
            url += f"&time_from={time_from}&time_to={time_to}"
            print(f"Fetching news for {symbol.upper()} on {date.strftime('%Y-%m-%d')} (from {time_from} to {time_to})")
        
        print(f"API URL: {url}")
        response = requests.get(url)
        data = response.json()
        
        if "feed" not in data:
            print(f"Error in news data response: {data}")
            if "Error Message" in data:
                print(f"API Error: {data['Error Message']}")
            if "Note" in data:
                print(f"API Note: {data['Note']}")
            return []
            
        articles = []
        print(f"Found {len(data['feed'])} news articles for {symbol.upper()}")
        
        for item in data["feed"][:10]:  # Limit to 10 articles
            # Extract published time for debugging
            published_time = item.get("time_published", "")
            if published_time:
                try:
                    # Format: YYYYMMDDTHHMM
                    pub_date = pd.Timestamp(
                        year=int(published_time[0:4]),
                        month=int(published_time[4:6]),
                        day=int(published_time[6:8]),
                        hour=int(published_time[9:11]),
                        minute=int(published_time[11:13])
                    )
                    formatted_date = pub_date.strftime('%Y-%m-%d %H:%M')
                except Exception as e:
                    formatted_date = published_time
                    print(f"Could not parse date {published_time}: {e}")
            else:
                formatted_date = "Unknown"
            
            # Safe extraction of ticker sentiment
            ticker_sentiment_data = {"ticker_sentiment_score": 0, "ticker_sentiment_label": "neutral"}
            for ticker_data in item.get("ticker_sentiment", []):
                if ticker_data.get("ticker") == symbol.upper():
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
                "publishedAt": published_time,
                "formatted_date": formatted_date,
                "url": item.get("url", ""),
                "description": item.get("summary", ""),
                "sentiment_score": overall_sentiment,
                "sentiment_label": item.get("overall_sentiment_label", "neutral"),
                "ticker_sentiment": ticker_sentiment_data
            }
            
            print(f"Article from {formatted_date}: {item.get('title', '')[:50]}... - Sentiment: {ticker_sentiment_data['ticker_sentiment_score']}")
            articles.append(article)
            
        return articles
        
    except Exception as e:
        print(f"Error fetching news from Alpha Vantage: {e}")
        traceback.print_exc()
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

def parse_sentiment_from_news(news_data):
    """
    Parse sentiment from news data.
    
    Args:
        news_data (list): List of news articles with sentiment data
        
    Returns:
        float: Average sentiment score
    """
    if not news_data:
        print("No news data to parse sentiment from")
        return 0
    
    sentiment_scores = []
    
    for article in news_data:
        # Get ticker-specific sentiment
        ticker_sentiment = article.get('ticker_sentiment', {})
        ticker_score = ticker_sentiment.get('ticker_sentiment_score', None)
        
        if ticker_score is not None:
            try:
                ticker_score = float(ticker_score)
                sentiment_scores.append(ticker_score)
                continue  # Use ticker-specific sentiment if available
            except (ValueError, TypeError):
                pass  # Fall back to overall sentiment if ticker sentiment is invalid
        
        # Fall back to overall sentiment
        overall_score = article.get('sentiment_score', None)
        if overall_score is not None:
            try:
                overall_score = float(overall_score)
                sentiment_scores.append(overall_score)
            except (ValueError, TypeError):
                pass  # Skip invalid overall sentiment
    
    if not sentiment_scores:
        print("No valid sentiment scores found in news data")
        return 0
    
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    print(f"Parsed {len(sentiment_scores)} sentiment scores, average: {avg_sentiment:.4f}")
    
    return avg_sentiment

def fetch_historical_sentiment_data(symbol, analysis_date=None):
    """
    Fetch historical sentiment data for a given symbol.
    
    Args:
        symbol (str): Stock symbol to fetch sentiment data for
        analysis_date (datetime, optional): The reference date for analysis. Defaults to current date.
    
    Returns:
        dict: Dictionary mapping dates to sentiment scores
    """
    # Convert symbol to uppercase
    symbol = symbol.upper()
    
    # Set analysis_date to current date if not provided
    if analysis_date is None:
        analysis_date = datetime.now()
    
    # Ensure analysis_date is a datetime object
    if isinstance(analysis_date, str):
        analysis_date = datetime.strptime(analysis_date, '%Y-%m-%d')
    
    # Get sentiment cache
    sentiment_cache = get_sentiment_cache()
    
    # If we already have data for this symbol, return it
    if symbol in sentiment_cache:
        return sentiment_cache[symbol]
    
    # Initialize sentiment data dictionary for this symbol
    sentiment_data = {}
    
    # Sample random dates from the past 200 days before the analysis date
    end_date = analysis_date
    start_date = end_date - timedelta(days=200)
    
    # Get 10 random dates in this range
    date_range = (end_date - start_date).days
    if date_range > 0:
        random_days = random.sample(range(date_range), min(10, date_range))
        sample_dates = [start_date + timedelta(days=day) for day in random_days]
        
        # Sort dates chronologically
        sample_dates.sort()
        
        # For each sample date, fetch sentiment data
        for date in sample_dates:
            date_str = date.strftime('%Y-%m-%d')
            try:
                # Get news data for the given date
                news_data = fetch_news_from_alpha_vantage(symbol, date)
                
                # Parse sentiment from news data
                sentiment = parse_sentiment_from_news(news_data)
                
                # Add sentiment to sentiment data
                sentiment_data[date_str] = sentiment
                
                # Wait to avoid rate limiting
                time.sleep(0.5)
            except Exception as e:
                print(f"Error fetching sentiment data for {symbol} on {date_str}: {str(e)}")
    else:
        print(f"Warning: Invalid date range for {symbol}. Using analysis date only.")
        # Just use the analysis date
        date_str = analysis_date.strftime('%Y-%m-%d')
        try:
            # Get news data for the analysis date
            news_data = fetch_news_from_alpha_vantage(symbol, analysis_date)
            
            # Parse sentiment from news data
            sentiment = parse_sentiment_from_news(news_data)
            
            # Add sentiment to sentiment data
            sentiment_data[date_str] = sentiment
        except Exception as e:
            print(f"Error fetching sentiment data for {symbol} on {date_str}: {str(e)}")
    
    # Cache sentiment data
    sentiment_cache[symbol] = sentiment_data
    save_sentiment_cache(sentiment_cache)
    
    return sentiment_data

def analyze_sentiment_correlation(symbol, historical_data, sentiment_data, analysis_date=None):
    """
    Analyze correlation between sentiment and price movements.
    
    :param symbol: Stock symbol
    :param historical_data: DataFrame of historical price data
    :param sentiment_data: Dictionary of sentiment scores by date
    :param analysis_date: Analysis date to use as reference (optional)
    :return: Dictionary with sentiment analysis results
    """
    # Create a DataFrame with sentiment and price data
    analysis_df = pd.DataFrame()
    
    # Add sentiment data
    sentiment_series = pd.Series(sentiment_data)
    analysis_df['sentiment'] = sentiment_series
    
    # Add price data from historical_data
    dates = []
    price_changes = []
    
    print(f"Analyzing correlation for {len(sentiment_data)} sentiment data points")
    
    # Count how many sentiment dates are found in historical data
    dates_found = 0
    
    for date_str, sentiment in sentiment_data.items():
        date = pd.Timestamp(date_str)
        if date in historical_data.index:
            dates_found += 1
            # Calculate if price went up or down that day
            try:
                open_price = historical_data.loc[date, 'Open']
                close_price = historical_data.loc[date, 'Close']
                price_change = (close_price - open_price) / open_price * 100
                
                dates.append(date_str)
                price_changes.append(price_change)
            except Exception as e:
                print(f"Error processing historical data for {date_str}: {e}")
    
    print(f"Found {dates_found} dates with both sentiment and price data")
    
    # If we don't have enough data points for correlation, use default values
    if len(dates) < 3:
        print(f"Insufficient data points ({len(dates)}) for correlation analysis")
        # Get current sentiment based on analysis date if provided
        if analysis_date:
            analysis_date_str = analysis_date.strftime('%Y-%m-%d')
            current_sentiment = sentiment_data.get(analysis_date_str, 0)
            if analysis_date_str not in sentiment_data:
                print(f"Analysis date {analysis_date_str} not found in sentiment data, using default sentiment of 0")
        else:
            # Fall back to the most recent sentiment data
            current_sentiment = list(sentiment_data.values())[-1] if sentiment_data else 0
        
        return {
            "current_sentiment": current_sentiment,
            "avg_sentiment_up": 0.1,  # Default positive sentiment
            "avg_sentiment_down": -0.1,  # Default negative sentiment
            "sentiment_prediction": "neutral",
            "confidence": 0,
            "up_days_count": 0,
            "down_days_count": 0,
            "correlation": 0
        }
    
    # Create DataFrame with aligned data
    corr_df = pd.DataFrame({
        'date': dates,
        'sentiment': [sentiment_data[d] for d in dates],
        'price_change': price_changes
    })
    
    # Separate into up days and down days
    up_days = corr_df[corr_df['price_change'] > 0]
    down_days = corr_df[corr_df['price_change'] <= 0]
    
    # Calculate average sentiment for up days and down days
    avg_sentiment_up = up_days['sentiment'].mean() if not up_days.empty else 0
    avg_sentiment_down = down_days['sentiment'].mean() if not down_days.empty else 0
    
    print(f"Average sentiment on up days: {avg_sentiment_up:.4f} ({len(up_days)} days)")
    print(f"Average sentiment on down days: {avg_sentiment_down:.4f} ({len(down_days)} days)")
    
    # Get current sentiment based on analysis date if provided
    if analysis_date:
        analysis_date_str = analysis_date.strftime('%Y-%m-%d')
        if analysis_date_str in sentiment_data:
            current_sentiment = sentiment_data[analysis_date_str]
            print(f"Using sentiment for analysis date {analysis_date_str}: {current_sentiment}")
        else:
            # If analysis date not in sentiment data, try to find closest date before it
            valid_dates = sorted([pd.Timestamp(d) for d in sentiment_data.keys()])
            closest_date = None
            for d in valid_dates:
                if d <= analysis_date:
                    closest_date = d
            
            if closest_date:
                closest_date_str = closest_date.strftime('%Y-%m-%d')
                current_sentiment = sentiment_data[closest_date_str]
                print(f"Analysis date {analysis_date_str} not found, using closest date {closest_date_str} with sentiment: {current_sentiment}")
            else:
                # Fall back to average sentiment
                current_sentiment = sum(sentiment_data.values()) / len(sentiment_data) if sentiment_data else 0
                print(f"No sentiment data found before analysis date, using average sentiment: {current_sentiment}")
    else:
        # If no analysis date, use the most recent sentiment
        try:
            current_sentiment = sentiment_data[max(sentiment_data.keys())]
        except (ValueError, KeyError):
            # If we can't get the latest sentiment, use the average
            current_sentiment = sum(sentiment_data.values()) / len(sentiment_data) if sentiment_data else 0
    
    # Determine if current sentiment is closer to up or down days
    sentiment_prediction = "neutral"
    if avg_sentiment_up > avg_sentiment_down:
        # If up days have higher sentiment than down days (expected pattern)
        if current_sentiment > (avg_sentiment_up + avg_sentiment_down) / 2:
            sentiment_prediction = "up"
        elif current_sentiment < (avg_sentiment_up + avg_sentiment_down) / 2:
            sentiment_prediction = "down"
    else:
        # Inverted pattern (higher sentiment on down days)
        if current_sentiment < (avg_sentiment_up + avg_sentiment_down) / 2:
            sentiment_prediction = "up"
        elif current_sentiment > (avg_sentiment_up + avg_sentiment_down) / 2:
            sentiment_prediction = "down"
    
    # Calculate confidence based on distance between average sentiments
    sentiment_gap = abs(avg_sentiment_up - avg_sentiment_down)
    confidence = min(100, max(0, sentiment_gap * 100)) if sentiment_gap > 0 else 0
    
    # Calculate correlation between sentiment and price changes
    try:
        correlation = corr_df['sentiment'].corr(corr_df['price_change']) if len(corr_df) > 5 else 0
    except Exception:
        correlation = 0
    
    return {
        "current_sentiment": current_sentiment,
        "avg_sentiment_up": avg_sentiment_up,
        "avg_sentiment_down": avg_sentiment_down,
        "sentiment_prediction": sentiment_prediction,
        "confidence": confidence,
        "up_days_count": len(up_days),
        "down_days_count": len(down_days),
        "correlation": correlation
    }

def get_sentiment_recommendation(symbol, historical_data, analysis_date=None):
    """
    Get a sentiment-based recommendation for a stock.
    
    :param symbol: Stock symbol
    :param historical_data: DataFrame of historical price data
    :param analysis_date: Analysis date to use as reference (optional)
    :return: Dictionary with sentiment recommendation
    """
    # Extract analysis date from historical data if not provided
    if analysis_date is None and not historical_data.empty:
        analysis_date = historical_data.index.max()
        print(f"Using latest historical data date as analysis date: {analysis_date}")
        
    # Fetch historical sentiment data
    sentiment_data = fetch_historical_sentiment_data(symbol, analysis_date=analysis_date)
    
    # Analyze correlation with price movements
    correlation_analysis = analyze_sentiment_correlation(symbol, historical_data, sentiment_data, analysis_date)
    
    # Generate recommendation
    recommendation = "NEUTRAL"
    if correlation_analysis["sentiment_prediction"] == "up" and correlation_analysis["confidence"] > 30:
        recommendation = "BUY"
    elif correlation_analysis["sentiment_prediction"] == "down" and correlation_analysis["confidence"] > 30:
        recommendation = "SELL"
    
    return {
        "recommendation": recommendation,
        "current_sentiment": correlation_analysis["current_sentiment"],
        "prediction": correlation_analysis["sentiment_prediction"],
        "confidence": correlation_analysis["confidence"],
        "correlation": correlation_analysis["correlation"],
        "analysis": correlation_analysis
    }

def clear_sentiment_cache_file():
    """
    Delete the sentiment cache file to force fresh data retrieval.
    """
    if os.path.exists(SENTIMENT_CACHE_FILE):
        try:
            os.remove(SENTIMENT_CACHE_FILE)
            print(f"Deleted sentiment cache file: {SENTIMENT_CACHE_FILE}")
            return True
        except Exception as e:
            print(f"Error deleting sentiment cache file: {e}")
            return False
    else:
        print(f"Sentiment cache file not found: {SENTIMENT_CACHE_FILE}")
        return False 