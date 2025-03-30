import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pandas_market_calendars as mcal

def backtest_strategy(df, analysis_date, stop_loss=0.05, take_profit=0.10):
    """
    Backtest the trading strategy and calculate 3-day and 5-day returns
    
    :param df: DataFrame with price data and indicators
    :param analysis_date: Date of analysis
    :param stop_loss: Stop loss percentage (default 5%)
    :param take_profit: Take profit percentage (default 10%)
    :return: Dictionary with backtest results
    """
    try:
        if df is None or df.empty:
            print("No data provided for backtesting")
            return None
            
        # Ensure we have enough data
        if len(df) < 20:  # Need at least 20 days for reliable backtesting
            print(f"Insufficient data points ({len(df)}) for backtesting")
            return None
        
        # Convert analysis_date to Timestamp if it's not already
        if not isinstance(analysis_date, pd.Timestamp):
            analysis_date = pd.Timestamp(analysis_date)
            
        # Get data after analysis date for backtesting
        future_data = df[df.index > pd.Timestamp(analysis_date)]
        if future_data.empty:
            print(f"No future data available for backtesting after {analysis_date}")
            return None
            
        # Get trading calendar for market days
        nyse = mcal.get_calendar('NYSE')
        
        # Get the next 10 trading days after analysis_date
        end_date = analysis_date + timedelta(days=20)  # Look ahead more days to ensure we have enough trading days
        trading_days = nyse.valid_days(start_date=analysis_date, end_date=end_date)
        
        print(f"Analysis date: {analysis_date}, Trading days available: {len(trading_days)}")
        
        # Get entry price (open price on analysis date - for buying at market open)
        entry_date = None
        entry_price = None
        
        # First check if the analysis date is in the dataframe
        if analysis_date in df.index:
            entry_date = analysis_date
            entry_price = df.loc[analysis_date, 'Open'] if 'Open' in df.columns else df.loc[analysis_date, 'Close']
        else:
            # Find the next trading day after analysis_date that exists in our data
            for day in trading_days:
                if day in df.index:
                    entry_date = day
                    entry_price = df.loc[day, 'Open'] if 'Open' in df.columns else df.loc[day, 'Close']
                    print(f"Analysis date {analysis_date} not found in data, using next trading day: {entry_date}")
                    break
        
        if entry_date is None or entry_price is None:
            print(f"Could not find a valid entry date/price after {analysis_date}")
            return None
            
        print(f"Entry date: {entry_date}, Entry price: ${entry_price:.2f}")
        
        # Calculate 3-day and 5-day returns based on trading days, not calendar days
        trading_days_in_data = [day for day in trading_days if day in df.index]
        entry_idx = trading_days_in_data.index(entry_date) if entry_date in trading_days_in_data else -1
        
        if entry_idx == -1:
            print(f"Entry date {entry_date} not found in trading calendar")
            return None
        
        # Calculate 3-day return (3rd trading day after entry)
        day3_date = None
        day3_price = None
        day3_return = None
        
        if entry_idx + 3 < len(trading_days_in_data):
            day3_date = trading_days_in_data[entry_idx + 3]
            if day3_date in df.index:
                day3_price = df.loc[day3_date, 'Close']
                day3_return = ((day3_price - entry_price) / entry_price) * 100
                print(f"3-day return date: {day3_date}, price: ${day3_price:.2f}, return: {day3_return:.2f}%")
            else:
                print(f"3-day date {day3_date} not found in data")
        else:
            print(f"Not enough trading days for 3-day return (need {entry_idx + 3}, have {len(trading_days_in_data)})")
        
        # Calculate 5-day return (5th trading day after entry)
        day5_date = None
        day5_price = None
        day5_return = None
        
        if entry_idx + 5 < len(trading_days_in_data):
            day5_date = trading_days_in_data[entry_idx + 5]
            if day5_date in df.index:
                day5_price = df.loc[day5_date, 'Close']
                day5_return = ((day5_price - entry_price) / entry_price) * 100
                print(f"5-day return date: {day5_date}, price: ${day5_price:.2f}, return: {day5_return:.2f}%")
            else:
                print(f"5-day date {day5_date} not found in data")
        else:
            print(f"Not enough trading days for 5-day return (need {entry_idx + 5}, have {len(trading_days_in_data)})")
        
        results = {
            'entry_price': entry_price,
            'entry_date': entry_date.strftime('%Y-%m-%d'),
            'day3_return': day3_return,
            'day3_date': day3_date.strftime('%Y-%m-%d') if day3_date is not None else None,
            'day3_price': day3_price,
            'day5_return': day5_return,
            'day5_date': day5_date.strftime('%Y-%m-%d') if day5_date is not None else None,
            'day5_price': day5_price
        }
        
        return results
        
    except Exception as e:
        print(f"Error in backtesting: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_win_rate(symbol, trends):
    """
    Calculate win rate based on stored trends.
    
    :param symbol: Stock symbol.
    :param trends: List of trend data.
    :return: Win rate statistics.
    """
    if not trends:
        return {"win_rate": 0, "total_trades": 0}
    
    # Count trades with positive returns
    wins = sum(1 for trend in trends if trend.get('direction') == 'up')
    
    # Calculate valid trades (ones with direction data)
    valid_trades = sum(1 for trend in trends if trend.get('direction') in ['up', 'down'])
    
    # Count stop loss and take profit triggers
    stop_loss_count = sum(1 for trend in trends if trend.get('reached_stop_loss', False))
    take_profit_count = sum(1 for trend in trends if trend.get('reached_take_profit', False))
    
    # Calculate win rate
    win_rate = (wins / valid_trades * 100) if valid_trades > 0 else 0
    
    # Calculate average return
    avg_return = sum(trend.get('actual_return', 0) for trend in trends if trend.get('actual_return') is not None) / valid_trades if valid_trades > 0 else 0
    
    # Calculate average missed opportunity
    missed_opps = [trend.get('max_potential_return', 0) - trend.get('actual_return', 0) 
                  for trend in trends 
                  if trend.get('max_potential_return') is not None and trend.get('actual_return') is not None]
    avg_missed_opportunity = sum(missed_opps) / len(missed_opps) if missed_opps else 0
    
    return {
        "win_rate": win_rate,
        "avg_return": avg_return,
        "total_trades": len(trends),
        "valid_trades": valid_trades,
        "stop_loss_triggered": stop_loss_count,
        "take_profit_triggered": take_profit_count,
        "stop_loss_percentage": (stop_loss_count / len(trends) * 100) if len(trends) > 0 else 0,
        "take_profit_percentage": (take_profit_count / len(trends) * 100) if len(trends) > 0 else 0,
        "avg_missed_opportunity": avg_missed_opportunity
    } 