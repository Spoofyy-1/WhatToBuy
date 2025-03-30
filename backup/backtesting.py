import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def backtest_strategy(df, analysis_date, market_df=None, stop_loss=0.05, take_profit=0.08):
    """
    Backtest a comprehensive investment strategy with stop loss and take profit.
    
    :param df: DataFrame containing historical stock data with indicators.
    :param analysis_date: Date of analysis as a datetime object.
    :param market_df: Optional market data for comparing relative performance.
    :param stop_loss: Stop loss percentage (default 5%)
    :param take_profit: Take profit percentage (default 8%)
    :return: Performance metrics of the strategy.
    """
    # Convert analysis_date to pandas Timestamp for comparison with DataFrame index
    analysis_date = pd.Timestamp(analysis_date)
    
    # Check if we have data for the analysis date
    if analysis_date not in df.index:
        # Find the closest available date
        available_dates = df.index[df.index <= analysis_date].tolist()
        if not available_dates:
            return {"error": "No data available before the specified date."}
        closest_date = max(available_dates)
        analysis_date = closest_date
        
    # Get the price on the analysis date
    try:
        price_on_date = df.loc[analysis_date, 'Close']
    except KeyError:
        return {"error": "No data available for the specified date or closest date."}
    
    # Get future dates for performance tracking
    future_dates = df.index[df.index > analysis_date].tolist()
    
    # Calculate performance metrics
    performance = {}
    performance['analysis_date'] = analysis_date.strftime('%Y-%m-%d')
    performance['price_on_date'] = price_on_date
    
    # Calculate stop loss and take profit prices
    stop_loss_price = price_on_date * (1 - stop_loss)
    take_profit_price = price_on_date * (1 + take_profit)
    
    # Track if stop loss or take profit was triggered
    performance['stop_loss_triggered'] = False
    performance['take_profit_triggered'] = False
    performance['exit_date'] = None
    performance['exit_price'] = None
    
    # Check 3-day performance with stop loss/take profit
    day3_date = None
    day3_price = None
    day3_return = None
    day3_direction = 'unknown'
    
    if len(future_dates) >= 1:
        for i, future_date in enumerate(future_dates[:min(5, len(future_dates))]):
            current_price = df.loc[future_date, 'Close']
            current_low = df.loc[future_date, 'Low']
            current_high = df.loc[future_date, 'High']
            
            # Check if stop loss triggered
            if current_low <= stop_loss_price:
                performance['stop_loss_triggered'] = True
                performance['exit_date'] = future_date.strftime('%Y-%m-%d')
                performance['exit_price'] = stop_loss_price
                day3_price = stop_loss_price
                day3_date = future_date
                break
                
            # Check if take profit triggered
            if current_high >= take_profit_price:
                performance['take_profit_triggered'] = True
                performance['exit_date'] = future_date.strftime('%Y-%m-%d')
                performance['exit_price'] = take_profit_price
                day3_price = take_profit_price
                day3_date = future_date
                break
                
            # If we've reached 3 days and no stop/take profit triggered
            if i == 2:  # 3rd day (index 2)
                day3_date = future_date
                day3_price = current_price
    
    # If exit wasn't triggered but we have 3-day data
    if not performance['exit_date'] and day3_date is not None:
        day3_price = df.loc[day3_date, 'Close']
        
    # Calculate 3-day return
    if day3_price is not None:
        day3_return = (day3_price - price_on_date) / price_on_date * 100
        day3_direction = 'up' if day3_return > 0 else 'down'
        
        performance['day3_return'] = day3_return
        performance['day3_direction'] = day3_direction
        
        # Compare with market (if available)
        if market_df is not None and day3_date in market_df.index and analysis_date in market_df.index:
            market_day3_return = (market_df.loc[day3_date, 'Close'] - market_df.loc[analysis_date, 'Close']) / market_df.loc[analysis_date, 'Close'] * 100
            performance['market_day3_return'] = market_day3_return
            performance['relative_day3_return'] = day3_return - market_day3_return
    else:
        performance['day3_return'] = None
        performance['day3_direction'] = 'unknown'
    
    # Check 5-day performance with stop loss/take profit
    day5_date = None
    day5_price = None
    day5_return = None
    day5_direction = 'unknown'
    
    # If exit already triggered, use that for 5-day performance too
    if performance['exit_date']:
        day5_date = datetime.strptime(performance['exit_date'], '%Y-%m-%d')
        day5_price = performance['exit_price']
    # Otherwise check if we have 5 days of data
    elif len(future_dates) >= 5:
        day5_date = future_dates[4]  # 5th day (index 4)
        day5_price = df.loc[day5_date, 'Close']
    
    # Calculate 5-day return
    if day5_price is not None:
        day5_return = (day5_price - price_on_date) / price_on_date * 100
        day5_direction = 'up' if day5_return > 0 else 'down'
        
        performance['day5_return'] = day5_return
        performance['day5_direction'] = day5_direction
        
        # Compare with market (if available)
        if market_df is not None and day5_date in market_df.index and analysis_date in market_df.index:
            market_day5_return = (market_df.loc[day5_date, 'Close'] - market_df.loc[analysis_date, 'Close']) / market_df.loc[analysis_date, 'Close'] * 100
            performance['market_day5_return'] = market_day5_return
            performance['relative_day5_return'] = day5_return - market_day5_return
    else:
        performance['day5_return'] = None
        performance['day5_direction'] = 'unknown'
    
    return performance

def calculate_win_rate(symbol, trends):
    """
    Calculate win rate based on stored trends.
    
    :param symbol: Stock symbol.
    :param trends: List of trend data.
    :return: Win rate statistics.
    """
    if not trends:
        return {"win_rate_3day": 0, "win_rate_5day": 0, "total_trades": 0}
    
    # Count trades with positive returns
    wins_3day = sum(1 for trend in trends if trend.get('day3_direction') == 'up')
    wins_5day = sum(1 for trend in trends if trend.get('day5_direction') == 'up')
    
    # Calculate valid trades (ones with 3-day and 5-day data)
    valid_trades_3day = sum(1 for trend in trends if trend.get('day3_direction') in ['up', 'down'])
    valid_trades_5day = sum(1 for trend in trends if trend.get('day5_direction') in ['up', 'down'])
    
    # Count stop loss and take profit triggers
    stop_loss_count = sum(1 for trend in trends if trend.get('stop_loss_triggered', False))
    take_profit_count = sum(1 for trend in trends if trend.get('take_profit_triggered', False))
    
    # Calculate win rates
    win_rate_3day = (wins_3day / valid_trades_3day * 100) if valid_trades_3day > 0 else 0
    win_rate_5day = (wins_5day / valid_trades_5day * 100) if valid_trades_5day > 0 else 0
    
    # Calculate average returns
    avg_return_3day = sum(trend.get('day3_return', 0) for trend in trends if trend.get('day3_return') is not None) / valid_trades_3day if valid_trades_3day > 0 else 0
    avg_return_5day = sum(trend.get('day5_return', 0) for trend in trends if trend.get('day5_return') is not None) / valid_trades_5day if valid_trades_5day > 0 else 0
    
    return {
        "win_rate_3day": win_rate_3day,
        "win_rate_5day": win_rate_5day,
        "avg_return_3day": avg_return_3day,
        "avg_return_5day": avg_return_5day,
        "total_trades": len(trends),
        "valid_trades_3day": valid_trades_3day,
        "valid_trades_5day": valid_trades_5day,
        "stop_loss_triggered": stop_loss_count,
        "take_profit_triggered": take_profit_count,
        "stop_loss_percentage": (stop_loss_count / len(trends) * 100) if len(trends) > 0 else 0,
        "take_profit_percentage": (take_profit_count / len(trends) * 100) if len(trends) > 0 else 0
    } 