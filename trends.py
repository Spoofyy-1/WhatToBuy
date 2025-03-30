import pandas as pd
import json
import os
from datetime import datetime

# File to store trends
TRENDS_FILE = 'stock_trends.json'

# Load existing trends from the JSON file
def load_trends():
    """
    Load existing trend data from JSON file.
    
    :return: Dictionary containing trend data for all stocks.
    """
    if os.path.exists(TRENDS_FILE):
        with open(TRENDS_FILE, 'r') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                print(f"Error reading {TRENDS_FILE}. Creating new trends database.")
                return {}
    return {}

# Save trends to the JSON file
def save_trends(trends):
    """
    Save trend data to JSON file with proper serialization.
    
    :param trends: Dictionary containing trend data for all stocks.
    """
    # Convert any non-serializable values (like numpy types) to standard Python types
    serializable_trends = {}
    for symbol, symbol_trends in trends.items():
        serializable_trends[symbol] = []
        for trend in symbol_trends:
            serializable_trend = {}
            for key, value in trend.items():
                # Convert numpy types to Python types
                if hasattr(value, 'item'):
                    serializable_trend[key] = value.item()
                # Handle nested dictionaries
                elif isinstance(value, dict):
                    serializable_nested = {}
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_value, 'item'):
                            serializable_nested[nested_key] = nested_value.item()
                        else:
                            serializable_nested[nested_key] = nested_value
                    serializable_trend[key] = serializable_nested
                else:
                    serializable_trend[key] = value
            serializable_trends[symbol].append(serializable_trend)
            
    with open(TRENDS_FILE, 'w') as file:
        json.dump(serializable_trends, file, indent=2)

# Initialize trends dictionary
stock_trends = load_trends()

def analyze_trend(symbol, historical_data, sentiment_score, signals, backtest_results, analysis_date):
    """
    Analyze trends for a specific stock based on historical data and sentiment score.
    
    :param symbol: Stock symbol to analyze.
    :param historical_data: DataFrame containing historical stock data.
    :param sentiment_score: Sentiment score from news articles.
    :param signals: Technical indicator signals.
    :param backtest_results: Results from backtesting.
    :param analysis_date: Date of analysis.
    :return: Trend analysis result.
    """
    if symbol not in stock_trends:
        stock_trends[symbol] = []

    # Ensure all values are properly typed
    try:
        actual_return = float(backtest_results.get('actual_return', 0)) if backtest_results and backtest_results.get('actual_return') is not None else None
        max_potential_return = float(backtest_results.get('max_potential_return', 0)) if backtest_results and backtest_results.get('max_potential_return') is not None else None
    except (ValueError, TypeError):
        actual_return = None
        max_potential_return = None
    
    # Convert analysis_date to Timestamp if it isn't already
    if not isinstance(analysis_date, pd.Timestamp):
        analysis_date = pd.Timestamp(analysis_date)

    # Get key technical values
    technical_values = {}
    
    # Extract all important technical indicators available
    if historical_data is not None and not historical_data.empty and analysis_date in historical_data.index:
        row = historical_data.loc[analysis_date]
        
        # Extract all important technical indicators available
        if 'Close' in row:
            technical_values['price'] = float(row['Close'])
        if 'RSI' in row:
            technical_values['rsi'] = float(row['RSI'])
        if 'MACD' in row:
            technical_values['macd'] = float(row['MACD'])
        if 'MACD_Signal' in row:
            technical_values['macd_signal'] = float(row['MACD_Signal'])
        if 'Volume_Ratio' in row:
            technical_values['volume_ratio'] = float(row['Volume_Ratio'])
        
        # Technical pattern indicators
        if 'SMA_20' in row and 'SMA_50' in row:
            technical_values['sma20_above_50'] = bool(row['SMA_20'] > row['SMA_50'])
        if 'SMA_50' in row and 'SMA_200' in row:
            technical_values['sma50_above_200'] = bool(row['SMA_50'] > row['SMA_200'])
        if 'Close' in row and 'SMA_20' in row:
            technical_values['price_above_sma20'] = bool(row['Close'] > row['SMA_20'])
        if 'Close' in row and 'SMA_50' in row:
            technical_values['price_above_sma50'] = bool(row['Close'] > row['SMA_50'])
        if 'Close' in row and 'SMA_200' in row:
            technical_values['price_above_sma200'] = bool(row['Close'] > row['SMA_200'])
        
        # Add Bollinger Band positions
        if all(x in row for x in ['Close', 'BB_Upper', 'BB_Lower']):
            if row['Close'] > row['BB_Upper']:
                technical_values['bollinger_position'] = 'above'
            elif row['Close'] < row['BB_Lower']:
                technical_values['bollinger_position'] = 'below'
            else:
                technical_values['bollinger_position'] = 'inside'
    
    # Create the signal summary from the signals parameter
    signal_summary = {}
    if signals:
        # Extract the most important signal values
        if 'overall_trend' in signals:
            signal_summary['overall_trend'] = float(signals['overall_trend'])
        if 'overall_momentum' in signals:
            signal_summary['overall_momentum'] = float(signals['overall_momentum'])
        
        # Add trend strength
        if 'trend_strength' in signals:
            signal_summary['trend_strength'] = {}
            if 'adx_strong' in signals['trend_strength']:
                signal_summary['trend_strength']['adx_strong'] = bool(signals['trend_strength']['adx_strong'])
            if 'adx_very_strong' in signals['trend_strength']:
                signal_summary['trend_strength']['adx_very_strong'] = bool(signals['trend_strength']['adx_very_strong'])
    
    # Market relative performance data
    market_relative = {}
    if backtest_results:
        if 'relative_performance' in backtest_results:
            market_relative['relative_return'] = float(backtest_results['relative_performance'])
        if 'market_return' in backtest_results:
            market_relative['market_return'] = float(backtest_results['market_return'])
    
    # Determine direction based on actual return
    direction = 'up' if actual_return and actual_return > 0 else 'down'
    
    # Create trend info with properly typed values and complete data
    trend_info = {
        'date': analysis_date.strftime('%Y-%m-%d'),
        'sentiment_score': float(sentiment_score),
        'actual_return': actual_return,
        'direction': direction,
        'max_potential_return': max_potential_return,
        'technical': technical_values,
        'signals': signal_summary,
        'market_relative': market_relative
    }
    
    # Add to trends
    stock_trends[symbol].append(trend_info)

    # Save updated trends to the JSON file
    save_trends(stock_trends)

    return stock_trends[symbol]

def get_trend(symbol):
    """
    Retrieve stored trends for a specific stock.
    
    :param symbol: Stock symbol to retrieve trends for.
    :return: List of trends for the stock.
    """
    return stock_trends.get(symbol, [])

def get_trend_insights(symbol):
    """
    Analyze stored trends to identify patterns and correlations.
    
    :param symbol: Stock symbol to analyze.
    :return: Dictionary of insights about the stock trends.
    """
    trends = get_trend(symbol)
    if not trends or len(trends) < 2:
        return {"message": "Not enough trend data to generate insights."}
    
    insights = {}
    
    # Analyze sentiment correlation with returns
    sentiment_return_corr = analyze_correlation(
        [t.get('sentiment_score', 0) for t in trends if 'sentiment_score' in t and 'actual_return' in t], 
        [t.get('actual_return', 0) for t in trends if 'sentiment_score' in t and 'actual_return' in t]
    )
    
    sentiment_max_return_corr = analyze_correlation(
        [t.get('sentiment_score', 0) for t in trends if 'sentiment_score' in t and 'max_potential_return' in t], 
        [t.get('max_potential_return', 0) for t in trends if 'sentiment_score' in t and 'max_potential_return' in t]
    )
    
    insights['sentiment_correlation'] = {
        'actual_return': sentiment_return_corr,
        'max_potential_return': sentiment_max_return_corr
    }
    
    # Analyze technical indicator effectiveness
    tech_insights = {}
    
    # Check RSI effectiveness
    rsi_trends = [t for t in trends if 'technical' in t and 'rsi' in t['technical'] and 'actual_return' in t]
    if rsi_trends:
        oversold_wins = sum(1 for t in rsi_trends if t['technical']['rsi'] < 30 and t['actual_return'] > 0)
        oversold_total = sum(1 for t in rsi_trends if t['technical']['rsi'] < 30)
        
        overbought_wins = sum(1 for t in rsi_trends if t['technical']['rsi'] > 70 and t['actual_return'] < 0)
        overbought_total = sum(1 for t in rsi_trends if t['technical']['rsi'] > 70)
        
        tech_insights['rsi'] = {
            'oversold_win_rate': (oversold_wins / oversold_total * 100) if oversold_total > 0 else None,
            'overbought_win_rate': (overbought_wins / overbought_total * 100) if overbought_total > 0 else None
        }
    
    # Check moving average effectiveness
    ma_trends = [t for t in trends if 'technical' in t and 'price_above_sma50' in t['technical'] and 'actual_return' in t]
    if ma_trends:
        above_sma50_wins = sum(1 for t in ma_trends if t['technical']['price_above_sma50'] and t['actual_return'] > 0)
        above_sma50_total = sum(1 for t in ma_trends if t['technical']['price_above_sma50'])
        
        below_sma50_wins = sum(1 for t in ma_trends if not t['technical']['price_above_sma50'] and t['actual_return'] < 0)
        below_sma50_total = sum(1 for t in ma_trends if not t['technical']['price_above_sma50'])
        
        tech_insights['moving_averages'] = {
            'above_sma50_win_rate': (above_sma50_wins / above_sma50_total * 100) if above_sma50_total > 0 else None,
            'below_sma50_win_rate': (below_sma50_wins / below_sma50_total * 100) if below_sma50_total > 0 else None
        }
    
    insights['technical_effectiveness'] = tech_insights
    
    return insights

def analyze_correlation(x_values, y_values):
    """
    Calculate correlation between two lists of values.
    
    :param x_values: First list of values.
    :param y_values: Second list of values.
    :return: Correlation coefficient.
    """
    if not x_values or not y_values or len(x_values) != len(y_values):
        return None
    
    n = len(x_values)
    if n < 2:
        return None
    
    # Calculate means
    x_mean = sum(x_values) / n
    y_mean = sum(y_values) / n
    
    # Calculate covariance and variances
    covariance = sum((x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n))
    x_variance = sum((x - x_mean) ** 2 for x in x_values)
    y_variance = sum((y - y_mean) ** 2 for y in y_values)
    
    # Calculate correlation coefficient
    if x_variance > 0 and y_variance > 0:
        correlation = covariance / ((x_variance * y_variance) ** 0.5)
        return correlation
    return None

def calculate_basket_win_rate(symbols):
    """
    Calculate combined win rate for a basket of stocks.
    
    :param symbols: List of stock symbols in the basket.
    :return: Win rate statistics for the entire basket.
    """
    all_trends = []
    for symbol in symbols:
        trends = get_trend(symbol)
        all_trends.extend(trends)
    
    if not all_trends:
        return {"win_rate": 0, "avg_return": 0, "total_trades": 0}
    
    # Count trades with positive returns
    wins = sum(1 for trend in all_trends if trend.get('direction') == 'up')
    
    # Calculate valid trades (ones with direction data)
    valid_trades = sum(1 for trend in all_trends if trend.get('direction') in ['up', 'down'])
    
    # Calculate win rates
    win_rate = (wins / valid_trades * 100) if valid_trades > 0 else 0
    
    # Calculate average returns
    avg_return = sum(trend.get('actual_return', 0) for trend in all_trends if trend.get('actual_return') is not None) / valid_trades if valid_trades > 0 else 0
    avg_max_return = sum(trend.get('max_potential_return', 0) for trend in all_trends if trend.get('max_potential_return') is not None) / valid_trades if valid_trades > 0 else 0
    
    return {
        "win_rate": win_rate,
        "avg_return": avg_return,
        "avg_max_return": avg_max_return,
        "total_trades": len(all_trends),
        "valid_trades": valid_trades
    } 