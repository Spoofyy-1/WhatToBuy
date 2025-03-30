import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

ALPHA_VANTAGE_API_KEY = 'YOUR API KEY'

def fetch_historical_data(symbol, start_date, end_date=None):
    """
    Fetch historical stock data from Alpha Vantage.
    
    :param symbol: Stock symbol to fetch data for.
    :param start_date: Start date for the data range
    :param end_date: End date for the data range
    :return: DataFrame containing historical stock data.
    """
    if end_date is None:
        end_date = start_date + timedelta(days=30)  # Extended to 30 days to ensure enough trading days
    
    print(f"Fetching data for {symbol} from {start_date} to {end_date}")
    
    # Try to use cached data first to avoid hitting API limits
    cache_dir = "data_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/{symbol}_data.csv"
    
    # Check if we have cached data
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index)
            cached_range = (df.index.min(), df.index.max())
            print(f"Loaded cached data for {symbol}. Date range: {cached_range[0]} to {cached_range[1]}")
            
            # Check if cached data covers our range
            if (df.index.min() <= pd.Timestamp(start_date) and 
                df.index.max() >= pd.Timestamp(end_date) and
                len(df) > 30):  # Ensure we have enough data points
                print(f"Using cached data for {symbol} (sufficient date range)")
                filtered_df = df[(df.index >= pd.Timestamp(start_date) - timedelta(days=10)) & 
                                (df.index <= pd.Timestamp(end_date) + timedelta(days=10))]
                return filtered_df
            else:
                print(f"Cached data for {symbol} insufficient. Need to fetch from API.")
        except Exception as e:
            print(f"Error reading cached data: {e}. Will fetch from API.")
    
    # Use full to get extensive historical data
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
    
    try:
        response = requests.get(url)
        
        # Check for rate limiting
        if "Note" in response.json() and "API call frequency" in response.json()["Note"]:
            print(f"API rate limit reached. Waiting 60 seconds before retrying...")
            time.sleep(60)  # Wait for rate limit to reset
            response = requests.get(url)  # Try again
        
        data = response.json()

        if "Time Series (Daily)" not in data:
            error_msg = data.get('Error Message', data.get('Information', 'Unknown error'))
            print(f"API Error: {error_msg}")
            
            # Try to fall back to cached data even if it's not ideal
            if os.path.exists(cache_file):
                print(f"Falling back to cached data for {symbol} due to API error")
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                df.index = pd.to_datetime(df.index)
                return df
                
            raise ValueError(f"Error fetching data for {symbol}: {error_msg}")

        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Rename columns
        df.columns = ['Open', 'High', 'Low', 'Close', 'Adjusted Close', 'Volume', 'Dividend Amount', 'Split Coefficient']
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Save to cache
        df.to_csv(cache_file)
        print(f"Saved data to cache: {cache_file}")
        
        print(f"Data fetched for {symbol}. Date range: {df.index.min()} to {df.index.max()}")
        
        # Filter to requested date range plus a buffer for calculations
        start_buffer = pd.Timestamp(start_date) - timedelta(days=10)
        end_buffer = pd.Timestamp(end_date) + timedelta(days=10)
        filtered_df = df[(df.index >= start_buffer) & (df.index <= end_buffer)]
        
        if filtered_df.empty:
            print(f"No data available in requested date range. Using available data.")
            # Find closest available date
            if pd.Timestamp(start_date) > df.index.max():
                print(f"Requested date {start_date} is after latest available data {df.index.max()}")
                # Use most recent available data with extra days before
                closest_date = df.index.max() - timedelta(days=30)
                filtered_df = df[df.index <= df.index.max()]
            elif pd.Timestamp(end_date) < df.index.min():
                print(f"Requested date {end_date} is before earliest available data {df.index.min()}")
                # Use earliest available data with extra days after
                closest_date = df.index.min() + timedelta(days=30)
                filtered_df = df[df.index >= df.index.min()]
            else:
                # Find data around the requested dates with a wider buffer
                available_dates = df.index.tolist()
                closest_date = min(available_dates, key=lambda x: abs(x - pd.Timestamp(start_date)))
                filtered_df = df[(df.index >= closest_date - timedelta(days=30)) & 
                                (df.index <= closest_date + timedelta(days=30))]
                
            print(f"Using closest available date: {closest_date} with extended buffer")
        
        return filtered_df
        
    except Exception as e:
        print(f"Exception in fetch_historical_data: {str(e)}")
        
        # Try to fall back to cached data as a last resort
        if os.path.exists(cache_file):
            print(f"Falling back to cached data for {symbol} due to exception")
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                df.index = pd.to_datetime(df.index)
                return df
            except:
                pass
                
        raise e

def fetch_market_data(analysis_date, force_refresh=False):
    """
    Fetch market index data (S&P 500) for context
    
    :param analysis_date: Analysis date
    :param force_refresh: Whether to bypass cache and get fresh data
    :return: DataFrame containing market data
    """
    # Use S&P 500 ETF (SPY) as market indicator
    try:
        start_date = analysis_date - timedelta(days=365)  # Look back one year
        end_date = analysis_date + timedelta(days=10)     # Look forward 10 days for backtesting
        
        # Use outputsize=full to get more historical data
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=SPY&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
        
        # Add timestamp to URL to prevent caching if force_refresh is True
        if force_refresh:
            url += f"&timestamp={datetime.now().timestamp()}"
        
        print(f"Fetching market data from: {url}")
        response = requests.get(url)
        data = response.json()

        if "Time Series (Daily)" not in data:
            print(f"Error fetching market data: {data.get('Error Message', data.get('Information', 'Unknown error'))}")
            return None

        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        df.columns = ['Open', 'High', 'Low', 'Close', 'Adjusted Close', 'Volume', 'Dividend Amount', 'Split Coefficient']
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Filter to requested date range
        filtered_df = df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]
        
        if filtered_df.empty:
            print("No market data available in requested date range")
            return None
            
        print(f"Market data fetched. Date range: {filtered_df.index.min()} to {filtered_df.index.max()}")
        return filtered_df
        
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return None

def get_stock_price_on_date(symbol, analysis_date):
    """
    Get the exact stock price on a specific date using Alpha Vantage.
    
    :param symbol: Stock symbol
    :param analysis_date: The specific date to get the price for
    :return: Dictionary with price data or None if not available
    """
    # Format the date for API call
    date_str = analysis_date.strftime('%Y-%m-%d')
    print(f"Getting exact price for {symbol} on {date_str} from Alpha Vantage")
    
    # We'll use the daily adjusted endpoint with full output size to ensure we get the specific date
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
    
    try:
        # Add a timestamp to prevent caching
        url += f"&timestamp={datetime.now().timestamp()}"
        
        response = requests.get(url)
        data = response.json()

        if "Time Series (Daily)" not in data:
            error_msg = data.get('Error Message', data.get('Information', 'Unknown error'))
            print(f"API Error getting price: {error_msg}")
            return None

        # Get the time series data
        time_series = data["Time Series (Daily)"]
        
        # Look for the exact date
        if date_str in time_series:
            price_data = time_series[date_str]
            result = {
                'date': date_str,
                'open': float(price_data['1. open']),
                'high': float(price_data['2. high']),
                'low': float(price_data['3. low']),
                'close': float(price_data['4. close']),
                'adjusted_close': float(price_data['5. adjusted close']),
                'volume': float(price_data['6. volume'])
            }
            print(f"Found exact price for {symbol} on {date_str}: ${result['close']}")
            return result
        
        # If exact date not found, find the nearest earlier date (trading day)
        print(f"Exact date {date_str} not found in data, finding nearest trading day")
        available_dates = sorted(time_series.keys())
        
        # Filter dates before or equal to our target date
        valid_dates = [d for d in available_dates if d <= date_str]
        
        if valid_dates:
            nearest_date = valid_dates[-1]  # Get the most recent date before target
            price_data = time_series[nearest_date]
            result = {
                'date': nearest_date,
                'open': float(price_data['1. open']),
                'high': float(price_data['2. high']),
                'low': float(price_data['3. low']),
                'close': float(price_data['4. close']),
                'adjusted_close': float(price_data['5. adjusted close']),
                'volume': float(price_data['6. volume']),
                'note': f"No data for {date_str}. Using closest trading day: {nearest_date}"
            }
            print(f"Using nearest date {nearest_date} with price: ${result['close']}")
            return result
        
        print(f"No valid trading dates found before {date_str}")
        return None
    
    except Exception as e:
        print(f"Error getting stock price for {symbol} on {date_str}: {str(e)}")
        return None 
