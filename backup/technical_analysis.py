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
    Calculate comprehensive technical indicators.
    
    :param df: DataFrame with stock price data
    :return: DataFrame with added technical indicators
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Moving Averages
    result['SMA_20'] = result['Close'].rolling(window=20).mean()
    result['SMA_50'] = result['Close'].rolling(window=50).mean()
    result['SMA_200'] = result['Close'].rolling(window=200).mean()
    result['EMA_9'] = result['Close'].ewm(span=9, adjust=False).mean()
    result['EMA_12'] = result['Close'].ewm(span=12, adjust=False).mean()
    result['EMA_26'] = result['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    result['MACD'] = result['EMA_12'] - result['EMA_26']
    result['MACD_Signal'] = result['MACD'].ewm(span=9, adjust=False).mean()
    result['MACD_Histogram'] = result['MACD'] - result['MACD_Signal']
    result['MACD_Crossover'] = ((result['MACD'] > result['MACD_Signal']) & 
                               (result['MACD'].shift(1) <= result['MACD_Signal'].shift(1))).astype(int)
    result['MACD_Crossunder'] = ((result['MACD'] < result['MACD_Signal']) & 
                                (result['MACD'].shift(1) >= result['MACD_Signal'].shift(1))).astype(int)
    
    # RSI
    result['RSI'] = calculate_rsi(result['Close'])
    
    # RSI divergence
    result['RSI_Higher_High'] = ((result['RSI'] > result['RSI'].shift(1)) & 
                                (result['RSI'].shift(1) > result['RSI'].shift(2))).astype(int)
    result['Price_Higher_High'] = ((result['Close'] > result['Close'].shift(1)) & 
                                 (result['Close'].shift(1) > result['Close'].shift(2))).astype(int)
    result['RSI_Bearish_Divergence'] = ((result['Price_Higher_High'] == 1) & 
                                       (result['RSI_Higher_High'] == 0)).astype(int)
    
    # Bollinger Bands
    result['BB_Middle'] = result['Close'].rolling(window=20).mean()
    result['BB_StdDev'] = result['Close'].rolling(window=20).std()
    result['BB_Upper'] = result['BB_Middle'] + (result['BB_StdDev'] * 2)
    result['BB_Lower'] = result['BB_Middle'] - (result['BB_StdDev'] * 2)
    result['BB_Width'] = (result['BB_Upper'] - result['BB_Lower']) / result['BB_Middle']
    result['BB_Squeeze'] = result['BB_Width'] < result['BB_Width'].rolling(window=50).mean()
    
    # Volume indicators
    result['Volume_SMA_20'] = result['Volume'].rolling(window=20).mean()
    result['Volume_Ratio'] = result['Volume'] / result['Volume_SMA_20']
    
    # Volume Price Trend
    vpt = (result['Volume'] * ((result['Close'] - result['Close'].shift(1)) / result['Close'].shift(1))).fillna(0)
    result['VPT'] = vpt.cumsum()
    
    # Volatility
    result['Daily_Return'] = result['Close'].pct_change()
    result['Volatility_20'] = result['Daily_Return'].rolling(window=20).std() * np.sqrt(252)  # Annualized
    
    # Momentum indicators
    result['ROC_5'] = result['Close'].pct_change(periods=5) * 100  # 5-day Rate of Change
    result['ROC_10'] = result['Close'].pct_change(periods=10) * 100
    
    # Stochastic Oscillator
    high_14 = result['High'].rolling(window=14).max()
    low_14 = result['Low'].rolling(window=14).min()
    result['Stoch_K'] = 100 * ((result['Close'] - low_14) / (high_14 - low_14))
    result['Stoch_D'] = result['Stoch_K'].rolling(window=3).mean()
    
    # ADX (Trend strength)
    result['ADX'] = calculate_adx(result)
    
    # Candlestick Patterns - Basic
    result['Engulfing_Bullish'] = calculate_bullish_engulfing(result)
    result['Engulfing_Bearish'] = calculate_bearish_engulfing(result)
    result['Doji'] = calculate_doji(result)
    
    # NEW: Enhanced Candlestick Patterns
    result['Hammer'] = calculate_hammer(result)
    result['Hanging_Man'] = calculate_hanging_man(result)
    result['Shooting_Star'] = calculate_shooting_star(result)
    result['Inverted_Hammer'] = calculate_inverted_hammer(result)
    result['Morning_Star'] = calculate_morning_star(result)
    result['Evening_Star'] = calculate_evening_star(result)
    result['Three_White_Soldiers'] = calculate_three_white_soldiers(result)
    result['Three_Black_Crows'] = calculate_three_black_crows(result)
    result['Harami_Bullish'] = calculate_bullish_harami(result)
    result['Harami_Bearish'] = calculate_bearish_harami(result)
    result['Piercing_Line'] = calculate_piercing_line(result)
    result['Dark_Cloud_Cover'] = calculate_dark_cloud_cover(result)
    result['Marubozu'] = calculate_marubozu(result)
    
    # Support and resistance detection
    result['Is_Support'] = detect_support(result)
    result['Is_Resistance'] = detect_resistance(result)
    
    return result

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

def analyze_indicators(df, date):
    """
    Analyze technical indicators for a specific date
    
    :param df: DataFrame with technical indicators
    :param date: Date to analyze
    :return: Dictionary of signals and their strengths
    """
    if date not in df.index:
        closest_date = df.index[df.index <= pd.Timestamp(date)].max()
        if pd.isnull(closest_date):
            return None
        date = closest_date
        
    row = df.loc[date]
    
    signals = {}
    
    # Trend signals
    signals['trend'] = {}
    signals['trend']['sma_20_above_50'] = 1 if row['SMA_20'] > row['SMA_50'] else -1
    signals['trend']['sma_50_above_200'] = 1 if row['SMA_50'] > row['SMA_200'] else -1
    signals['trend']['price_above_sma_20'] = 1 if row['Close'] > row['SMA_20'] else -1
    signals['trend']['price_above_sma_50'] = 1 if row['Close'] > row['SMA_50'] else -1
    signals['trend']['price_above_sma_200'] = 1 if row['Close'] > row['SMA_200'] else -1
    signals['trend']['ema_9_above_20'] = 1 if row['EMA_9'] > row['SMA_20'] else -1
    
    # Momentum signals
    signals['momentum'] = {}
    signals['momentum']['macd_above_signal'] = 1 if row['MACD'] > row['MACD_Signal'] else -1
    signals['momentum']['macd_positive'] = 1 if row['MACD'] > 0 else -1
    signals['momentum']['macd_crossover'] = 2 if row['MACD_Crossover'] == 1 else 0
    signals['momentum']['macd_crossunder'] = -2 if row['MACD_Crossunder'] == 1 else 0
    signals['momentum']['rsi_overbought'] = -2 if row['RSI'] > 70 else 0
    signals['momentum']['rsi_oversold'] = 2 if row['RSI'] < 30 else 0
    signals['momentum']['rsi_trending_up'] = 1 if row['RSI'] > 50 else -1
    signals['momentum']['stoch_k_above_d'] = 1 if row['Stoch_K'] > row['Stoch_D'] else -1
    signals['momentum']['stoch_oversold'] = 2 if row['Stoch_K'] < 20 else 0
    signals['momentum']['stoch_overbought'] = -2 if row['Stoch_K'] > 80 else 0
    
    # Volatility signals
    signals['volatility'] = {}
    signals['volatility']['bollinger_breakout_upper'] = -1 if row['Close'] > row['BB_Upper'] else 0
    signals['volatility']['bollinger_breakout_lower'] = 1 if row['Close'] < row['BB_Lower'] else 0
    signals['volatility']['bollinger_squeeze'] = 1 if row['BB_Squeeze'] else 0
    signals['volatility']['volatility_high'] = -1 if row['Volatility_20'] > df['Volatility_20'].quantile(0.8) else 0
    
    # Volume signals
    signals['volume'] = {}
    signals['volume']['above_average'] = 1 if row['Volume_Ratio'] > 1.5 else (-1 if row['Volume_Ratio'] < 0.5 else 0)
    signals['volume']['rising_volume'] = 1 if row['Volume'] > df['Volume'].shift(1).loc[date] else -1
    signals['volume']['vpt_rising'] = 1 if row['VPT'] > df['VPT'].shift(1).loc[date] else -1
    
    # Pattern signals - Basic
    signals['patterns'] = {}
    signals['patterns']['bullish_engulfing'] = 2 if row['Engulfing_Bullish'] == 1 else 0
    signals['patterns']['bearish_engulfing'] = -2 if row['Engulfing_Bearish'] == 1 else 0
    signals['patterns']['doji'] = 1 if row['Doji'] == 1 else 0
    signals['patterns']['at_support'] = 2 if row['Is_Support'] == 1 else 0
    signals['patterns']['at_resistance'] = -2 if row['Is_Resistance'] == 1 else 0
    signals['patterns']['bearish_divergence'] = -2 if row['RSI_Bearish_Divergence'] == 1 else 0
    
    # NEW: Enhanced Candlestick Pattern Signals
    signals['candlestick_patterns'] = {}
    
    # Single-candle patterns
    signals['candlestick_patterns']['hammer'] = 2 if row['Hammer'] == 1 else 0
    signals['candlestick_patterns']['hanging_man'] = -2 if row['Hanging_Man'] == 1 else 0
    signals['candlestick_patterns']['shooting_star'] = -2 if row['Shooting_Star'] == 1 else 0
    signals['candlestick_patterns']['inverted_hammer'] = 2 if row['Inverted_Hammer'] == 1 else 0
    signals['candlestick_patterns']['marubozu'] = 2 if row['Marubozu'] == 1 and row['Close'] > row['Open'] else (-2 if row['Marubozu'] == 1 else 0)
    
    # Multi-candle patterns (stronger signals)
    signals['candlestick_patterns']['morning_star'] = 3 if row['Morning_Star'] == 1 else 0
    signals['candlestick_patterns']['evening_star'] = -3 if row['Evening_Star'] == 1 else 0
    signals['candlestick_patterns']['three_white_soldiers'] = 3 if row['Three_White_Soldiers'] == 1 else 0
    signals['candlestick_patterns']['three_black_crows'] = -3 if row['Three_Black_Crows'] == 1 else 0
    signals['candlestick_patterns']['harami_bullish'] = 2 if row['Harami_Bullish'] == 1 else 0
    signals['candlestick_patterns']['harami_bearish'] = -2 if row['Harami_Bearish'] == 1 else 0
    signals['candlestick_patterns']['piercing_line'] = 2 if row['Piercing_Line'] == 1 else 0
    signals['candlestick_patterns']['dark_cloud_cover'] = -2 if row['Dark_Cloud_Cover'] == 1 else 0
    
    # Overall trend strength
    signals['trend_strength'] = {}
    signals['trend_strength']['adx_strong'] = 1 if row['ADX'] > 25 else 0
    signals['trend_strength']['adx_very_strong'] = 1 if row['ADX'] > 40 else 0
    
    # Calculate market condition based on volatility and trend
    if row['Volatility_20'] > df['Volatility_20'].quantile(0.8):
        signals['market_condition'] = 'volatile'
    elif (row['SMA_50'] > row['SMA_200']) and (row['Close'] > row['SMA_50']):
        signals['market_condition'] = 'bullish'
    elif (row['SMA_50'] < row['SMA_200']) and (row['Close'] < row['SMA_50']):
        signals['market_condition'] = 'bearish'
    else:
        signals['market_condition'] = 'neutral'
    
    # Aggregated signals
    signals['overall_trend'] = (
        signals['trend']['sma_20_above_50'] +
        signals['trend']['sma_50_above_200'] * 2 +
        signals['trend']['price_above_sma_20'] +
        2 * signals['trend']['price_above_sma_50'] +
        3 * signals['trend']['price_above_sma_200'] +
        signals['trend']['ema_9_above_20']
    ) / 10  # Normalize to -1 to 1 range
    
    signals['overall_momentum'] = (
        2 * signals['momentum']['macd_above_signal'] +
        signals['momentum']['macd_positive'] +
        signals['momentum']['macd_crossover'] +
        signals['momentum']['macd_crossunder'] +
        signals['momentum']['rsi_overbought'] +
        signals['momentum']['rsi_oversold'] +
        signals['momentum']['rsi_trending_up'] +
        signals['momentum']['stoch_k_above_d'] +
        signals['momentum']['stoch_oversold'] +
        signals['momentum']['stoch_overbought']
    ) / 12  # Normalize
    
    signals['overall_volume'] = (
        signals['volume']['above_average'] * 2 +
        signals['volume']['rising_volume'] +
        signals['volume']['vpt_rising']
    ) / 4  # Normalize
    
    # NEW: Add enhanced candlestick patterns to the mix
    candlestick_score = sum(signals['candlestick_patterns'].values()) / 20  # Normalize
    
    # Update patterns score to include candlestick patterns
    basic_patterns_score = (
        signals['patterns']['bullish_engulfing'] +
        signals['patterns']['bearish_engulfing'] +
        signals['patterns']['doji'] +
        signals['patterns']['at_support'] +
        signals['patterns']['at_resistance'] +
        signals['patterns']['bearish_divergence']
    ) / 10  # Normalize
    
    signals['overall_patterns'] = (basic_patterns_score + candlestick_score) / 2
    
    return signals 