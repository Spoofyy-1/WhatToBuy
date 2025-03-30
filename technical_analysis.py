import pandas as pd
import numpy as np

# Add this at the top to make pattern detection safer
def safe_compare(left, right, operator='<'):
    """Safely compare values, handling NaN values and type mismatches"""
    if isinstance(left, pd.Series) or isinstance(right, pd.Series):
        if operator == '<':
            return pd.Series(left < right).fillna(False)
        elif operator == '>':
            return pd.Series(left > right).fillna(False)
        elif operator == '<=':
            return pd.Series(left <= right).fillna(False)
        elif operator == '>=':
            return pd.Series(left >= right).fillna(False)
        elif operator == '==':
            return pd.Series(left == right).fillna(False)
    return False

# Monkey patch pandas Series to handle NaN in boolean operations
original_and = pd.Series.__and__
def safe_and(left, right):
    """
    Safely perform logical AND between two values that might be NaN or pandas Series.
    """
    # Handle pandas Series objects
    if isinstance(left, pd.Series):
        if left.empty:
            return False
        # If it's a Series with one element, extract the scalar value
        if len(left) == 1:
            left = left.iloc[0]
        else:
            # For multi-element Series, decide how to handle it
            return left.all() and (right.all() if isinstance(right, pd.Series) else right)
    
    # Handle pandas Series for right operand
    if isinstance(right, pd.Series):
        if right.empty:
            return False
        # If it's a Series with one element, extract the scalar value
        if len(right) == 1:
            right = right.iloc[0]
        else:
            # For multi-element Series, decide how to handle it
            return (left.all() if isinstance(left, pd.Series) else left) and right.all()
    
    # Handle NaN values
    if isinstance(left, float) and pd.isna(left):
        return False
    if isinstance(right, float) and pd.isna(right):
        return False
    
    # Cast to boolean explicitly for regular values
    return bool(left) and bool(right)

pd.Series.__and__ = safe_and

def safe_condition(condition):
    """
    Handle NaN values and pandas Series in conditions
    """
    if isinstance(condition, pd.Series):
        if condition.empty:
            return False
        # If it's a Series with one element, extract the scalar value
        if len(condition) == 1:
            condition = condition.iloc[0]
        else:
            # For multi-element Series, decide how to handle it
            return condition.all()  # or condition.any() depending on your needs
    
    if isinstance(condition, float) and pd.isna(condition):
        return False
    
    return bool(condition)

def calculate_indicators(df):
    """
    Calculate technical indicators for stock analysis
    
    :param df: DataFrame with OHLCV data
    :return: Dictionary of technical indicators
    """
    try:
        if df is None or df.empty:
            print("No data provided for technical analysis")
            return None
            
        # Ensure we have enough data for calculations
        if len(df) < 20:  # Need at least 20 days for basic indicators
            print(f"Insufficient data points ({len(df)}) for technical analysis")
            return None
            
        # Calculate SMAs
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        
        # Calculate MACD (12,26,9)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Calculate RSI with 14 periods
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate Volume Ratio (20-day)
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
        
        # Get latest values
        latest = df.iloc[-1]
        prev_day = df.iloc[-2] if len(df) > 1 else latest
        
        # Calculate price position relative to SMAs
        price = latest['Close']
        sma20 = latest['SMA20']
        sma50 = latest['SMA50']
        sma200 = latest['SMA200']
        
        # Calculate trend strengths
        sma20_trend = ((price - sma20) / sma20) * 100 if pd.notnull(sma20) else 0
        sma50_trend = ((price - sma50) / sma50) * 100 if pd.notnull(sma50) else 0
        sma200_trend = ((price - sma200) / sma200) * 100 if pd.notnull(sma200) else 0
        
        # MACD signals
        macd = latest['MACD']
        signal = latest['Signal_Line']
        prev_macd = prev_day['MACD']
        prev_signal = prev_day['Signal_Line']
        macd_crossover = (macd > signal and prev_macd <= prev_signal)
        macd_crossunder = (macd < signal and prev_macd >= prev_signal)
        
        indicators = {
            'price': price,
            'sma20': sma20,
            'sma50': sma50,
            'sma200': sma200,
            'price_above_sma20': price > sma20 if pd.notnull(sma20) else None,
            'price_above_sma50': price > sma50 if pd.notnull(sma50) else None,
            'price_above_sma200': price > sma200 if pd.notnull(sma200) else None,
            'sma20_trend': sma20_trend,
            'sma50_trend': sma50_trend,
            'sma200_trend': sma200_trend,
            'macd': macd,
            'macd_signal': signal,
            'macd_histogram': macd - signal,
            'macd_crossover': macd_crossover,
            'macd_crossunder': macd_crossunder,
            'rsi': latest['RSI'],
            'volume_ratio': latest['Volume_Ratio'],
            'volume': latest['Volume'],
            'volume_ma20': latest['Volume_MA20']
        }
        
        # Add trend analysis
        indicators['short_term_trend'] = 'up' if sma20_trend > 0 else 'down'
        indicators['medium_term_trend'] = 'up' if sma50_trend > 0 else 'down'
        indicators['long_term_trend'] = 'up' if sma200_trend > 0 else 'down'
        
        # Calculate overall trend score (-100 to 100)
        trend_score = (
            (sma20_trend * 0.5) +  # Short term (50% weight)
            (sma50_trend * 0.3) +  # Medium term (30% weight)
            (sma200_trend * 0.2)   # Long term (20% weight)
        )
        indicators['trend_score'] = trend_score
        
        return indicators
        
    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
        return None

def detect_support(df, window=5, threshold=0.01):
    """Detect support levels based on previous lows"""
    is_support = pd.Series(False, index=df.index)
    
    for i in range(window, len(df)-window):
        if all(df['Low'].iloc[i] < df['Low'].iloc[i-j] for j in range(1, window+1)) and \
           all(df['Low'].iloc[i] < df['Low'].iloc[i+j] for j in range(1, window+1)):
            is_support.iloc[i] = True
    
    return is_support.astype(int)

def detect_resistance(df, window=5, threshold=0.01):
    """Detect resistance levels based on previous highs"""
    is_resistance = pd.Series(False, index=df.index)
    
    for i in range(window, len(df)-window):
        if all(df['High'].iloc[i] > df['High'].iloc[i-j] for j in range(1, window+1)) and \
           all(df['High'].iloc[i] > df['High'].iloc[i+j] for j in range(1, window+1)):
            is_resistance.iloc[i] = True
    
    return is_resistance.astype(int)

def calculate_bullish_engulfing(df):
    """
    Detect bullish engulfing candlestick pattern.
    """
    # Instead, create a new Series with the calculation and return that:
    result = pd.Series(False, index=df.index)
    
    for i in range(1, len(df)):
        # Get single values for current and previous candles
        curr_open = df['Open'].iloc[i]
        curr_close = df['Close'].iloc[i]
        prev_open = df['Open'].iloc[i-1]
        prev_close = df['Close'].iloc[i-1]
        
        # Check if current candle engulfs previous candle
        is_bullish_engulfing = (curr_close > curr_open and  # Current candle is bullish
                               curr_open <= prev_close and  # Current open is below or equal to previous close
                               curr_close > prev_open)      # Current close is above previous open
        
        result.iloc[i] = is_bullish_engulfing
    
    return result

def calculate_bearish_engulfing(df):
    """Detect bearish engulfing patterns"""
    bearish_engulfing = ((df['Open'] > df['Close'].shift(1)) & 
                         (df['Close'] < df['Open'].shift(1)) &
                         (df['Open'].shift(1) < df['Close'].shift(1)))
    return bearish_engulfing.astype(int)

def calculate_doji(df, threshold=0.1):
    """Detect doji patterns (open and close very close)"""
    body_size = abs(df['Close'] - df['Open'])
    candle_range = df['High'] - df['Low']
    doji = body_size < (threshold * candle_range)
    return doji.astype(int)

# NEW CANDLESTICK PATTERN FUNCTIONS

def calculate_hammer(df, body_ratio=0.3, wick_ratio=2.0):
    """
    Detect hammer patterns (small body near top with long lower wick)
    Bullish reversal signal in downtrends
    """
    # Calculate body and wick sizes
    body_size = abs(df['Close'] - df['Open'])
    total_range = df['High'] - df['Low']
    lower_wick = df[['Open', 'Close']].min(axis=1) - df['Low']
    upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)
    
    # Make sure SMA_20 exists to avoid errors
    if 'SMA_20' not in df.columns:
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Use .fillna() to handle NaN values before boolean comparisons
    cond1 = (body_size < body_ratio * total_range).fillna(False)
    cond2 = (lower_wick > wick_ratio * body_size).fillna(False)
    cond3 = (upper_wick < 0.3 * body_size).fillna(False)
    cond4 = (df['Close'] < df['SMA_20']).fillna(False)
    
    # Identify hammers using ratios
    is_hammer = cond1 & cond2 & cond3 & cond4
    
    return is_hammer.astype(int)

def calculate_hanging_man(df, body_ratio=0.3, wick_ratio=2.0):
    """
    Detect hanging man patterns (small body near top with long lower wick)
    Bearish reversal signal in uptrends
    """
    # Calculate body and wick sizes
    body_size = abs(df['Close'] - df['Open'])
    total_range = df['High'] - df['Low']
    lower_wick = df[['Open', 'Close']].min(axis=1) - df['Low']
    upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)
    
    # Make sure SMA_20 exists to avoid errors
    if 'SMA_20' not in df.columns:
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Use .fillna() to handle NaN values before boolean comparisons
    cond1 = safe_condition(body_size < body_ratio * total_range)
    cond2 = safe_condition(lower_wick > wick_ratio * body_size)
    cond3 = safe_condition(upper_wick < 0.3 * body_size)
    cond4 = safe_condition(df['Close'] > df['SMA_20'])
    
    # Identify hanging man using ratios
    is_hanging_man = cond1 & cond2 & cond3 & cond4
    
    return is_hanging_man.astype(int)

def calculate_shooting_star(df, body_ratio=0.3, wick_ratio=2.0):
    try:
        # Original code with safe conversion
        body_size = abs(df['Close'] - df['Open'])
        total_range = df['High'] - df['Low']
        lower_wick = df[['Open', 'Close']].min(axis=1) - df['Low']
        upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)
        
        # Make sure SMA_20 exists
        if 'SMA_20' not in df.columns:
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        cond1 = safe_condition(body_size < body_ratio * total_range)
        cond2 = safe_condition(upper_wick > wick_ratio * body_size)
        cond3 = safe_condition(lower_wick < 0.3 * body_size)
        cond4 = safe_condition(df['Close'] > df['SMA_20'])
        
        is_shooting_star = cond1 & cond2 & cond3 & cond4
        
        return is_shooting_star.astype(int)
    except Exception as e:
        # Return a series of zeros on error
        return pd.Series(0, index=df.index)

def calculate_inverted_hammer(df, body_ratio=0.3, wick_ratio=2.0):
    """
    Detect inverted hammer patterns (small body near bottom with long upper wick)
    Bullish reversal signal in downtrends
    """
    # Calculate body and wick sizes
    body_size = abs(df['Close'] - df['Open'])
    total_range = df['High'] - df['Low']
    lower_wick = df[['Open', 'Close']].min(axis=1) - df['Low']
    upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)
    
    # Make sure SMA_20 exists to avoid errors
    if 'SMA_20' not in df.columns:
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Use .fillna() to handle NaN values before boolean comparisons
    cond1 = safe_condition(body_size < body_ratio * total_range)
    cond2 = safe_condition(upper_wick > wick_ratio * body_size)
    cond3 = safe_condition(lower_wick < 0.3 * body_size)
    cond4 = safe_condition(df['Close'] < df['SMA_20'])
    
    # Identify inverted hammers using ratios
    is_inverted_hammer = cond1 & cond2 & cond3 & cond4
    
    return is_inverted_hammer.astype(int)

def calculate_morning_star(df, doji_threshold=0.1):
    """
    Detect morning star patterns (three-candle bullish reversal pattern)
    First day: long bearish candle
    Second day: small body (doji-like) gapping down
    Third day: bullish candle closing into first candle's body
    """
    # First day: bearish candle
    first_bearish = (df['Open'].shift(2) > df['Close'].shift(2))
    first_body_size = abs(df['Open'].shift(2) - df['Close'].shift(2))
    
    # Second day: small body (doji-like) with gap down
    second_small_body = abs(df['Open'].shift(1) - df['Close'].shift(1)) < (doji_threshold * (df['High'].shift(1) - df['Low'].shift(1)))
    gap_down = df[['Open', 'Close']].shift(1).max(axis=1) < df['Close'].shift(2)
    
    # Third day: bullish candle closing into first candle's body
    third_bullish = (df['Close'] > df['Open'])
    close_into_first = (df['Close'] > (df['Open'].shift(2) + df['Close'].shift(2)) / 2)
    
    # Combined pattern
    is_morning_star = (
        first_bearish &
        first_body_size > 0.01 * df['Close'].shift(2) &  # Ensure first candle is significant
        second_small_body &
        third_bullish &
        close_into_first
    )
    
    return is_morning_star.astype(int)

def calculate_evening_star(df, doji_threshold=0.1):
    """
    Detect evening star patterns (three-candle bearish reversal pattern)
    First day: long bullish candle
    Second day: small body (doji-like) gapping up
    Third day: bearish candle closing into first candle's body
    """
    # First day: bullish candle
    first_bullish = (df['Close'].shift(2) > df['Open'].shift(2))
    first_body_size = abs(df['Open'].shift(2) - df['Close'].shift(2))
    
    # Second day: small body (doji-like) with gap up
    second_small_body = abs(df['Open'].shift(1) - df['Close'].shift(1)) < (doji_threshold * (df['High'].shift(1) - df['Low'].shift(1)))
    gap_up = df[['Open', 'Close']].shift(1).min(axis=1) > df['Close'].shift(2)
    
    # Third day: bearish candle closing into first candle's body
    third_bearish = (df['Close'] < df['Open'])
    close_into_first = (df['Close'] < (df['Open'].shift(2) + df['Close'].shift(2)) / 2)
    
    # Combined pattern
    is_evening_star = (
        first_bullish &
        first_body_size > 0.01 * df['Close'].shift(2) &  # Ensure first candle is significant
        second_small_body &
        third_bearish &
        close_into_first
    )
    
    return is_evening_star.astype(int)

def calculate_three_white_soldiers(df, min_body_ratio=0.6):
    """
    Detect three white soldiers pattern (three consecutive bullish candles with higher highs)
    Strong bullish continuation pattern
    """
    # Three consecutive bullish candles
    bullish_candles = (df['Close'] > df['Open']) & (df['Close'].shift(1) > df['Open'].shift(1)) & (df['Close'].shift(2) > df['Open'].shift(2))
    
    # Each candle opens within previous candle's body and closes higher
    proper_sequence = (
        (df['Open'] > df['Open'].shift(1)) & (df['Open'] < df['Close'].shift(1)) &
        (df['Open'].shift(1) > df['Open'].shift(2)) & (df['Open'].shift(1) < df['Close'].shift(2)) &
        (df['Close'] > df['Close'].shift(1)) & (df['Close'].shift(1) > df['Close'].shift(2))
    )
    
    # Each candle has substantial body
    substantial_bodies = (
        (abs(df['Close'] - df['Open']) > min_body_ratio * (df['High'] - df['Low'])) &
        (abs(df['Close'].shift(1) - df['Open'].shift(1)) > min_body_ratio * (df['High'].shift(1) - df['Low'].shift(1))) &
        (abs(df['Close'].shift(2) - df['Open'].shift(2)) > min_body_ratio * (df['High'].shift(2) - df['Low'].shift(2)))
    )
    
    # Combined pattern
    is_three_white_soldiers = bullish_candles & proper_sequence & substantial_bodies
    
    return is_three_white_soldiers.astype(int)

def calculate_three_black_crows(df, min_body_ratio=0.6):
    """
    Detect three black crows pattern (three consecutive bearish candles with lower lows)
    Strong bearish continuation pattern
    """
    # Three consecutive bearish candles
    bearish_candles = (df['Close'] < df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1)) & (df['Close'].shift(2) < df['Open'].shift(2))
    
    # Each candle opens within previous candle's body and closes lower
    proper_sequence = (
        (df['Open'] < df['Open'].shift(1)) & (df['Open'] > df['Close'].shift(1)) &
        (df['Open'].shift(1) < df['Open'].shift(2)) & (df['Open'].shift(1) > df['Close'].shift(2)) &
        (df['Close'] < df['Close'].shift(1)) & (df['Close'].shift(1) < df['Close'].shift(2))
    )
    
    # Each candle has substantial body
    substantial_bodies = (
        (abs(df['Close'] - df['Open']) > min_body_ratio * (df['High'] - df['Low'])) &
        (abs(df['Close'].shift(1) - df['Open'].shift(1)) > min_body_ratio * (df['High'].shift(1) - df['Low'].shift(1))) &
        (abs(df['Close'].shift(2) - df['Open'].shift(2)) > min_body_ratio * (df['High'].shift(2) - df['Low'].shift(2)))
    )
    
    # Combined pattern
    is_three_black_crows = bearish_candles & proper_sequence & substantial_bodies
    
    return is_three_black_crows.astype(int)

def calculate_bullish_harami(df):
    """
    Detect bullish harami patterns (small bullish candle contained within previous larger bearish candle)
    Bullish reversal signal
    """
    # Previous candle is bearish and current candle is bullish
    prev_bearish_curr_bullish = (df['Open'].shift(1) > df['Close'].shift(1)) & (df['Close'] > df['Open'])
    
    # Current candle is contained within previous candle's body
    contained_body = (df['Open'] > df['Close'].shift(1)) & (df['Close'] < df['Open'].shift(1))
    
    # Previous candle has substantial body
    prev_substantial = abs(df['Open'].shift(1) - df['Close'].shift(1)) > 0.5 * (df['High'].shift(1) - df['Low'].shift(1))
    
    # Combined pattern
    is_bullish_harami = prev_bearish_curr_bullish & contained_body & prev_substantial
    
    return is_bullish_harami.astype(int)

def calculate_bearish_harami(df):
    """
    Detect bearish harami patterns (small bearish candle contained within previous larger bullish candle)
    Bearish reversal signal
    """
    # Previous candle is bullish and current candle is bearish
    prev_bullish_curr_bearish = (df['Close'].shift(1) > df['Open'].shift(1)) & (df['Open'] > df['Close'])
    
    # Current candle is contained within previous candle's body
    contained_body = (df['Open'] < df['Close'].shift(1)) & (df['Close'] > df['Open'].shift(1))
    
    # Previous candle has substantial body
    prev_substantial = abs(df['Open'].shift(1) - df['Close'].shift(1)) > 0.5 * (df['High'].shift(1) - df['Low'].shift(1))
    
    # Combined pattern
    is_bearish_harami = prev_bullish_curr_bearish & contained_body & prev_substantial
    
    return is_bearish_harami.astype(int)

def calculate_piercing_line(df, threshold=0.5):
    """
    Detect piercing line patterns (bearish candle followed by bullish candle that closes above midpoint of previous candle)
    Bullish reversal signal
    """
    # Previous candle is bearish
    prev_bearish = df['Open'].shift(1) > df['Close'].shift(1)
    
    # Current candle is bullish
    curr_bullish = df['Close'] > df['Open']
    
    # Current candle opens below previous close and closes above midpoint of previous candle
    piercing = (
        (df['Open'] < df['Close'].shift(1)) &
        (df['Close'] > df['Open'].shift(1) + threshold * (df['Close'].shift(1) - df['Open'].shift(1)))
    )
    
    # Combined pattern
    is_piercing_line = prev_bearish & curr_bullish & piercing
    
    return is_piercing_line.astype(int)

def calculate_dark_cloud_cover(df, threshold=0.5):
    """
    Detect dark cloud cover patterns (bullish candle followed by bearish candle that closes below midpoint of previous candle)
    Bearish reversal signal
    """
    # Previous candle is bullish
    prev_bullish = df['Close'].shift(1) > df['Open'].shift(1)
    
    # Current candle is bearish
    curr_bearish = df['Open'] > df['Close']
    
    # Current candle opens above previous close and closes below midpoint of previous candle
    cloud_cover = (
        (df['Open'] > df['Close'].shift(1)) &
        (df['Close'] < df['Open'].shift(1) + threshold * (df['Close'].shift(1) - df['Open'].shift(1)))
    )
    
    # Combined pattern
    is_dark_cloud_cover = prev_bullish & curr_bearish & cloud_cover
    
    return is_dark_cloud_cover.astype(int)

def calculate_marubozu(df, body_ratio=0.9):
    """
    Detect marubozu patterns (candles with very little or no wicks)
    Strong trend signal
    """
    # Calculate body size relative to candle range
    body_size = abs(df['Close'] - df['Open'])
    candle_range = df['High'] - df['Low']
    
    # Bullish marubozu (close at high, open at low)
    bullish_marubozu = (
        (df['Close'] > df['Open']) &
        (df['High'] - df['Close'] < 0.1 * body_size) &
        (df['Open'] - df['Low'] < 0.1 * body_size)
    )
    
    # Bearish marubozu (open at high, close at low)
    bearish_marubozu = (
        (df['Open'] > df['Close']) &
        (df['High'] - df['Open'] < 0.1 * body_size) &
        (df['Close'] - df['Low'] < 0.1 * body_size)
    )
    
    # Combined pattern (either type)
    is_marubozu = (bullish_marubozu | bearish_marubozu) & (body_size > body_ratio * candle_range)
    
    return is_marubozu.astype(int)

def calculate_adx(df, period=14):
    """Calculate Average Directional Index"""
    high_copy = df['High'].copy()
    low_copy = df['Low'].copy()
    close_copy = df['Close'].copy()
    
    # True Range
    tr1 = abs(high_copy - low_copy)
    tr2 = abs(high_copy - close_copy.shift())
    tr3 = abs(low_copy - close_copy.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    # Plus Directional Movement (+DM)
    plus_dm = high_copy.diff()
    minus_dm = low_copy.diff().multiply(-1)
    plus_dm = pd.Series(np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0), index=df.index)
    minus_dm = pd.Series(np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0), index=df.index)
    
    # Smooth +DM and -DM using Wilder's smoothing method
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    
    # Calculate Directional Index (DX)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # Calculate ADX as the moving average of DX
    adx = dx.rolling(period).mean()
    
    return adx

def calculate_rsi(price_series, window=14):
    """
    Calculate the Relative Strength Index (RSI) for a price series
    
    Parameters:
    -----------
    price_series : pandas.Series
        Series of prices
    window : int, default 14
        The lookback period for RSI calculation
        
    Returns:
    --------
    pandas.Series : RSI values
    """
    # Make sure we're working with a Series
    if not isinstance(price_series, pd.Series):
        price_series = pd.Series(price_series)
    
    # Calculate price changes
    delta = price_series.diff()
    
    # Create positive and negative change Series
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss over the window
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def analyze_indicators(df, date, technical_indicators=None):
    """
    Analyze technical indicators for a specific date
    
    :param df: DataFrame with price data
    :param date: Date to analyze
    :param technical_indicators: Optional dictionary of pre-calculated technical indicators
    :return: Dictionary of signals and their strengths
    """
    # If technical indicators are provided, use them directly
    if technical_indicators is not None:
        signals = {}
        
        # Trend signals
        signals['trend'] = {}
        signals['trend']['sma_20_above_50'] = 1 if technical_indicators.get('sma20', 0) > technical_indicators.get('sma50', 0) else -1
        signals['trend']['sma_50_above_200'] = 1 if technical_indicators.get('sma50', 0) > technical_indicators.get('sma200', 0) else -1
        signals['trend']['price_above_sma_20'] = 1 if technical_indicators.get('price_above_sma20', False) else -1
        signals['trend']['price_above_sma_50'] = 1 if technical_indicators.get('price_above_sma50', False) else -1
        signals['trend']['price_above_sma_200'] = 1 if technical_indicators.get('price_above_sma200', False) else -1
        
        # Momentum signals
        signals['momentum'] = {}
        signals['momentum']['macd_above_signal'] = 1 if technical_indicators.get('macd', 0) > technical_indicators.get('macd_signal', 0) else -1
        signals['momentum']['macd_positive'] = 1 if technical_indicators.get('macd', 0) > 0 else -1
        signals['momentum']['macd_crossover'] = 2 if technical_indicators.get('macd_crossover', False) else 0
        signals['momentum']['macd_crossunder'] = -2 if technical_indicators.get('macd_crossunder', False) else 0
        signals['momentum']['rsi_overbought'] = -2 if technical_indicators.get('rsi', 0) > 70 else 0
        signals['momentum']['rsi_oversold'] = 2 if technical_indicators.get('rsi', 0) < 30 else 0
        signals['momentum']['rsi_trending_up'] = 1 if technical_indicators.get('rsi', 0) > 50 else -1
        
        # Volume signals
        signals['volume'] = {}
        signals['volume']['above_average'] = 1 if technical_indicators.get('volume_ratio', 0) > 1.2 else -1
        
        # Calculate overall signals
        trend_sum = sum(signals['trend'].values())
        trend_count = len(signals['trend'])
        momentum_sum = sum(signals['momentum'].values())
        momentum_count = len(signals['momentum'])
        volume_sum = sum(signals['volume'].values())
        volume_count = len(signals['volume'])
        
        signals['overall_trend'] = trend_sum / trend_count if trend_count > 0 else 0
        signals['overall_momentum'] = momentum_sum / momentum_count if momentum_count > 0 else 0
        signals['overall_volume'] = volume_sum / volume_count if volume_count > 0 else 0
        
        # Trend strength
        signals['trend_strength'] = {
            'adx_strong': technical_indicators.get('adx', 0) > 25,
            'adx_very_strong': technical_indicators.get('adx', 0) > 40
        }
        
        # Market condition
        if signals['overall_trend'] > 0.5 and signals['overall_momentum'] > 0.5:
            signals['market_condition'] = 'bullish'
        elif signals['overall_trend'] < -0.5 and signals['overall_momentum'] < -0.5:
            signals['market_condition'] = 'bearish'
        else:
            signals['market_condition'] = 'neutral'
            
        return signals
    
    # Otherwise, extract signals from the DataFrame
    if date not in df.index:
        closest_date = df.index[df.index <= pd.Timestamp(date)].max()
        if pd.isnull(closest_date):
            return None
        date = closest_date
        
    # Check if we have the necessary columns
    required_columns = ['SMA20', 'SMA50', 'SMA200', 'MACD', 'Signal_Line', 'RSI', 'Volume', 'Volume_MA20']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        # Calculate indicators if they're missing
        df = calculate_indicators(df)
        if df is None:
            return None
    
    row = df.loc[date]
    
    signals = {}
    
    # Trend signals
    signals['trend'] = {}
    signals['trend']['sma_20_above_50'] = 1 if row['SMA20'] > row['SMA50'] else -1
    signals['trend']['sma_50_above_200'] = 1 if row['SMA50'] > row['SMA200'] else -1
    signals['trend']['price_above_sma_20'] = 1 if row['Close'] > row['SMA20'] else -1
    signals['trend']['price_above_sma_50'] = 1 if row['Close'] > row['SMA50'] else -1
    signals['trend']['price_above_sma_200'] = 1 if row['Close'] > row['SMA200'] else -1
    
    # Momentum signals
    signals['momentum'] = {}
    signals['momentum']['macd_above_signal'] = 1 if row['MACD'] > row['Signal_Line'] else -1
    signals['momentum']['macd_positive'] = 1 if row['MACD'] > 0 else -1
    signals['momentum']['macd_crossover'] = 2 if row['macd_crossover'] == 1 else 0
    signals['momentum']['macd_crossunder'] = -2 if row['macd_crossunder'] == 1 else 0
    signals['momentum']['rsi_overbought'] = -2 if row['RSI'] > 70 else 0
    signals['momentum']['rsi_oversold'] = 2 if row['RSI'] < 30 else 0
    signals['momentum']['rsi_trending_up'] = 1 if row['RSI'] > 50 else -1
    
    # Volume signals
    signals['volume'] = {}
    signals['volume']['above_average'] = 1 if row['volume_ratio'] > 1.2 else -1
    
    # Calculate overall signals
    trend_sum = sum(signals['trend'].values())
    trend_count = len(signals['trend'])
    momentum_sum = sum(signals['momentum'].values())
    momentum_count = len(signals['momentum'])
    volume_sum = sum(signals['volume'].values())
    volume_count = len(signals['volume'])
    
    signals['overall_trend'] = trend_sum / trend_count if trend_count > 0 else 0
    signals['overall_momentum'] = momentum_sum / momentum_count if momentum_count > 0 else 0
    signals['overall_volume'] = volume_sum / volume_count if volume_count > 0 else 0
    
    # Trend strength
    signals['trend_strength'] = {
        'adx_strong': calculate_adx(df) > 25,
        'adx_very_strong': calculate_adx(df) > 40
    }
    
    # Market condition
    if signals['overall_trend'] > 0.5 and signals['overall_momentum'] > 0.5:
        signals['market_condition'] = 'bullish'
    elif signals['overall_trend'] < -0.5 and signals['overall_momentum'] < -0.5:
        signals['market_condition'] = 'bearish'
    else:
        signals['market_condition'] = 'neutral'
    
    return signals 