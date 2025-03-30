import pandas as pd
import calendar
import os
from datetime import datetime, timedelta
from data_fetch import fetch_historical_data, fetch_market_data
from technical_analysis import calculate_indicators, analyze_indicators
from backtesting import backtest_strategy, calculate_win_rate
from sentiment_analysis import fetch_news_from_alpha_vantage, analyze_sentiment
from trends import analyze_trend, get_trend, calculate_basket_win_rate

def generate_recommendation(signals, sentiment_score, historical_performance=None):
    """
    Generate investment recommendation based on technical signals and sentiment.
    
    :param signals: Dictionary of technical signals
    :param sentiment_score: Sentiment score from news
    :param historical_performance: Optional historical performance metrics
    :return: Recommendation and reasoning
    """
    # Initialize score components
    technical_score = 0
    sentiment_score_weighted = sentiment_score * 3  # Scale sentiment to similar range as other signals
    
    # Calculate technical score from signals
    if signals:
        # Trend components (weighted higher for longer-term signals)
        technical_score += signals['trend']['price_above_sma_20'] * 1
        technical_score += signals['trend']['price_above_sma_50'] * 2
        technical_score += signals['trend']['price_above_sma_200'] * 3
        technical_score += signals['trend']['sma_20_above_50'] * 1
        technical_score += signals['trend']['sma_50_above_200'] * 2
        
        # Momentum components
        technical_score += signals['momentum']['macd_above_signal'] * 2
        technical_score += signals['momentum']['macd_positive'] * 1
        technical_score -= signals['momentum']['rsi_overbought'] * 2  # Overbought is negative
        technical_score += signals['momentum']['rsi_oversold'] * 2  # Oversold is positive for buying
        
        # Volume component
        if 'volume' in signals:
            technical_score += signals['volume']['above_average'] * 1
            technical_score += signals['volume'].get('rising_volume', 0) * 1
            technical_score += signals['volume'].get('vpt_rising', 0) * 1
        
        # Pattern signals (give these more weight as they're specific)
        if 'patterns' in signals:
            technical_score += signals['patterns'].get('bullish_engulfing', 0) * 2
            technical_score += signals['patterns'].get('bearish_engulfing', 0) * 2
            technical_score += signals['patterns'].get('at_support', 0) * 2
            technical_score += signals['patterns'].get('at_resistance', 0) * 2
        
        # Trend strength - strong trends are more reliable
        if 'trend_strength' in signals:
            if signals['trend_strength'].get('adx_strong', 0):
                technical_score *= 1.2
            if signals['trend_strength'].get('adx_very_strong', 0):
                technical_score *= 1.3
        
        # Normalize to range of -10 to +10
        technical_score = max(min(technical_score, 10), -10)
    
    # Combined score (70% technical, 30% sentiment)
    combined_score = (technical_score * 0.7) + (sentiment_score_weighted * 0.3)
    
    # Generate recommendation
    recommendation = "HOLD"  # Default
    reasoning = []
    
    if combined_score > 3:
        recommendation = "STRONG BUY"
        reasoning.append("Very positive technical indicators and sentiment")
    elif combined_score > 1:
        recommendation = "BUY"
        reasoning.append("Positive technical indicators and sentiment")
    elif combined_score < -3:
        recommendation = "STRONG SELL"
        reasoning.append("Very negative technical indicators and sentiment")
    elif combined_score < -1:
        recommendation = "SELL"
        reasoning.append("Negative technical indicators and sentiment")
    
    # Add specific reasoning based on strongest signals
    if signals:
        if 'overall_trend' in signals and signals['overall_trend'] > 0.5:
            reasoning.append("Strong upward price trend across multiple timeframes")
        elif 'overall_trend' in signals and signals['overall_trend'] < -0.5:
            reasoning.append("Strong downward price trend across multiple timeframes")
            
        if 'momentum' in signals:
            if signals['momentum'].get('macd_crossover', 0) > 0:
                reasoning.append("Recent MACD bullish crossover")
            elif signals['momentum'].get('macd_crossunder', 0) < 0:
                reasoning.append("Recent MACD bearish crossunder")
            
            if signals['momentum'].get('rsi_overbought', 0) != 0:
                reasoning.append("RSI indicates overbought conditions (potential reversal)")
            elif signals['momentum'].get('rsi_oversold', 0) != 0:
                reasoning.append("RSI indicates oversold conditions (potential buying opportunity)")
        
        if 'patterns' in signals:
            if signals['patterns'].get('bullish_engulfing', 0) > 0:
                reasoning.append("Bullish engulfing pattern detected")
            if signals['patterns'].get('bearish_engulfing', 0) < 0:
                reasoning.append("Bearish engulfing pattern detected")
            if signals['patterns'].get('at_support', 0) > 0:
                reasoning.append("Price at support level")
            if signals['patterns'].get('at_resistance', 0) < 0:
                reasoning.append("Price at resistance level")
            
    if sentiment_score > 0.2:
        reasoning.append("Very positive news sentiment")
    elif sentiment_score > 0.05:
        reasoning.append("Positive news sentiment")
    elif sentiment_score < -0.2:
        reasoning.append("Very negative news sentiment")
    elif sentiment_score < -0.05:
        reasoning.append("Negative news sentiment")
    
    return {
        "recommendation": recommendation,
        "score": combined_score,
        "technical_score": technical_score,
        "sentiment_score": sentiment_score,
        "reasoning": reasoning
    }

def allocate_investment(total_amount, results, market_condition=None):
    """
    Allocate investment amount among stocks based on recommendation scores.
    
    :param total_amount: Total amount to invest
    :param results: List of analysis results for each stock
    :param market_condition: Optional market condition for risk management
    :return: Dictionary mapping symbols to investment amounts
    """
    allocation = {}
    valid_results = [r for r in results if "recommendation" in r and "error" not in r]
    
    if not valid_results:
        return allocation
    
    # If in a bearish market, consider allocating less total capital
    risk_factor = 1.0  # Default risk factor
    if market_condition == 'bearish':
        risk_factor = 0.5  # Reduce position sizes in bearish markets
    elif market_condition == 'volatile':
        risk_factor = 0.7  # Reduce position sizes in volatile markets
    
    adjusted_total = total_amount * risk_factor
    
    # Extract scores and normalize them
    scores = []
    for result in valid_results:
        score = result["recommendation"]["score"]
        
        # Add a minimum score threshold for investing
        min_score_threshold = 2.0  # Only invest in strong BUY signals
        
        # Check if any stocks have bullish patterns
        has_bullish_pattern = False
        if "signals" in result and "patterns" in result["signals"]:
            if result["signals"]["patterns"].get("bullish_engulfing", 0) > 0 or \
               result["signals"]["patterns"].get("at_support", 0) > 0:
                has_bullish_pattern = True
        
        # Give bonus points for bullish patterns
        pattern_bonus = 2 if has_bullish_pattern else 0
        
        # Convert scores to a 0-100 scale, with higher being better
        normalized_score = (score + pattern_bonus + 10) / 20 * 100
        
        # Only invest in stocks with scores above threshold
        if score > min_score_threshold:
            scores.append((result["symbol"], max(normalized_score, 0)))
        else:
            scores.append((result["symbol"], 0))
    
    # Calculate total score
    total_score = sum(score for _, score in scores)
    
    # Allocate proportionally to scores
    if total_score > 0:
        for symbol, score in scores:
            allocation[symbol] = (score / total_score) * adjusted_total
    else:
        # If no stocks meet criteria, don't invest
        return {}
    
    # Apply per-stock risk limits
    max_per_stock = total_amount * 0.35  # No more than 35% in any one stock
    for symbol in allocation:
        if allocation[symbol] > max_per_stock:
            allocation[symbol] = max_per_stock
    
    return allocation

def analyze_stock(symbol, analysis_date, stop_loss=0.05, take_profit=0.08):
    """
    Analyze a single stock and return results.
    
    :param symbol: Stock symbol to analyze
    :param analysis_date: Date to analyze
    :param stop_loss: Stop loss percentage (default 5%)
    :param take_profit: Take profit percentage (default 8%)
    :return: Dictionary with analysis results
    """
    results = {"symbol": symbol, "date": analysis_date.strftime('%Y-%m-%d')}
    
    try:
        # Fetch historical data
        end_date = analysis_date + timedelta(days=10)
        historical_data = fetch_historical_data(symbol, analysis_date, end_date)
        
        # Fetch market data for context
        market_data = fetch_market_data(analysis_date)
        
        # Calculate technical indicators
        historical_data = calculate_indicators(historical_data)
        if market_data is not None:
            market_data = calculate_indicators(market_data)
            
        # Check if data is available
        if historical_data.empty:
            results["error"] = f"No data available for {symbol} around {analysis_date.strftime('%Y-%m-%d')}."
            return results

        # Find closest available date to analysis_date
        available_dates = historical_data.index.tolist()
        closest_date = min(available_dates, key=lambda x: abs(x - pd.Timestamp(analysis_date)))
        
        if closest_date != pd.Timestamp(analysis_date):
            results["note"] = f"Data not available for exact date. Using closest date: {closest_date.strftime('%Y-%m-%d')}"
            analysis_date = closest_date
        
        # Analyze technical indicators
        signals = analyze_indicators(historical_data, analysis_date)
        results["signals"] = signals
        
        # Get market condition
        if 'market_condition' in signals:
            results["market_condition"] = signals["market_condition"]
        
        # Fetch news and analyze sentiment using Alpha Vantage
        try:
            news_articles = fetch_news_from_alpha_vantage(symbol)
            sentiment_score = analyze_sentiment(news_articles)
            results["sentiment_score"] = sentiment_score
        except Exception as e:
            results["sentiment_warning"] = f"Error fetching news: {str(e)}"
            sentiment_score = 0

        # Generate recommendation
        recommendation_result = generate_recommendation(signals, sentiment_score)
        results["recommendation"] = recommendation_result
        
        # Get key technical indicators
        results["technical_indicators"] = {
            "price": historical_data.loc[analysis_date, 'Close'],
            "rsi": historical_data.loc[analysis_date, 'RSI'],
            "macd": historical_data.loc[analysis_date, 'MACD'],
            "macd_signal": historical_data.loc[analysis_date, 'MACD_Signal'],
            "sma_20": historical_data.loc[analysis_date, 'SMA_20'],
            "sma_50": historical_data.loc[analysis_date, 'SMA_50'],
            "sma_200": historical_data.loc[analysis_date, 'SMA_200'],
        }
        
        # Run backtesting (with stop loss and take profit)
        backtest_results = backtest_strategy(historical_data, analysis_date, market_data, 
                                           stop_loss=stop_loss, take_profit=take_profit)
        results["backtest"] = backtest_results
        
        # Store trend data
        analyze_trend(symbol, historical_data, sentiment_score, signals, backtest_results, analysis_date)
        
        # Calculate win rates
        win_rates = calculate_win_rate(symbol, get_trend(symbol))
        results["win_rates"] = win_rates
        
        return results
        
    except Exception as e:
        results["error"] = str(e)
        return results

def calculate_portfolio_performance(allocation, results, stop_loss=0.05, take_profit=0.08):
    """
    Calculate the expected performance of the portfolio based on backtesting.
    
    :param allocation: Dictionary mapping symbols to investment amounts
    :param results: List of analysis results for each stock
    :param stop_loss: Stop loss percentage
    :param take_profit: Take profit percentage
    :return: Dictionary with portfolio performance metrics
    """
    portfolio = {
        "initial_investment": sum(allocation.values()),
        "3day_value": 0,
        "5day_value": 0,
        "3day_return_pct": 0,
        "5day_return_pct": 0,
        "stocks": [],
        "stop_loss": stop_loss,
        "take_profit": take_profit
    }
    
    total_3day_value = 0
    total_5day_value = 0
    total_stop_loss_triggered = 0
    total_take_profit_triggered = 0
    
    for result in results:
        symbol = result["symbol"]
        if symbol not in allocation or "error" in result or "backtest" not in result:
            continue
            
        investment = allocation[symbol]
        backtest = result["backtest"]
        
        # Track whether stop loss or take profit was triggered
        stop_loss_triggered = backtest.get("stop_loss_triggered", False)
        take_profit_triggered = backtest.get("take_profit_triggered", False)
        
        if stop_loss_triggered:
            total_stop_loss_triggered += 1
        if take_profit_triggered:
            total_take_profit_triggered += 1
        
        # Calculate 3-day and 5-day values
        day3_return = backtest.get("day3_return", 0)
        day5_return = backtest.get("day5_return", 0)
        
        day3_value = investment * (1 + day3_return / 100)
        day5_value = investment * (1 + day5_return / 100)
        
        total_3day_value += day3_value
        total_5day_value += day5_value
        
        # Store individual stock performance
        portfolio["stocks"].append({
            "symbol": symbol,
            "investment": investment,
            "3day_value": day3_value,
            "5day_value": day5_value,
            "3day_return_pct": day3_return,
            "5day_return_pct": day5_return,
            "stop_loss_triggered": stop_loss_triggered,
            "take_profit_triggered": take_profit_triggered
        })
    
    # Calculate overall portfolio returns
    initial_investment = portfolio["initial_investment"]
    if initial_investment > 0:
        portfolio["3day_value"] = total_3day_value
        portfolio["5day_value"] = total_5day_value
        portfolio["3day_return_pct"] = (total_3day_value - initial_investment) / initial_investment * 100
        portfolio["5day_return_pct"] = (total_5day_value - initial_investment) / initial_investment * 100
        portfolio["stop_loss_count"] = total_stop_loss_triggered
        portfolio["take_profit_count"] = total_take_profit_triggered
    
    return portfolio

def main():
    print("Welcome to the Stock Investing Tool!")
    
    # User input for stock symbols and date
    symbols_input = input("Enter stock symbol(s) separated by space (e.g., AAPL MSFT AMZN): ").strip().upper()
    date_input = input("Enter a date to analyze (YYYY-MM-DD): ")

    # Parse stock symbols
    symbols = symbols_input.split()
    if not symbols:
        print("No stock symbols entered. Please try again.")
        return
    
    # Get investment amount
    try:
        if len(symbols) > 1:
            investment_input = input(f"Enter total amount to invest across all {len(symbols)} stocks ($): ")
        else:
            investment_input = input(f"Enter amount to invest in {symbols[0]} ($): ")
        total_investment = float(investment_input.replace('$', '').replace(',', ''))
        if total_investment <= 0:
            raise ValueError("Investment amount must be positive")
    except ValueError as e:
        print(f"Invalid investment amount: {e}")
        return
    
    # Get risk management preferences
    try:
        stop_loss_input = input("Enter stop loss percentage (default 5%): ")
        take_profit_input = input("Enter take profit percentage (default 8%): ")
        
        stop_loss = float(stop_loss_input) / 100 if stop_loss_input.strip() else 0.05
        take_profit = float(take_profit_input) / 100 if take_profit_input.strip() else 0.08
        
        if stop_loss <= 0 or stop_loss >= 1:
            print("Invalid stop loss. Using default value of 5%")
            stop_loss = 0.05
        
        if take_profit <= 0 or take_profit >= 1:
            print("Invalid take profit. Using default value of 8%")
            take_profit = 0.08
    except ValueError:
        print("Invalid risk parameters. Using defaults: 5% stop loss, 8% take profit")
        stop_loss = 0.05
        take_profit = 0.08
        
    # Convert input date to datetime object and check if it's in the future
    try:
        analysis_date = datetime.strptime(date_input, "%Y-%m-%d")
        
        # Check if date is in the future
        if analysis_date > datetime.now():
            print("The date cannot be in the future. Please enter a valid date.")
            return
            
        # Check if date is a weekend
        weekday = analysis_date.weekday()
        if weekday == 5 or weekday == 6:  # 5 is Saturday, 6 is Sunday
            print(f"The date {date_input} is a {calendar.day_name[weekday]}. Stock markets are closed on weekends.")
            print("Please enter a weekday date (Monday through Friday).")
            return
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD.")
        return

    # Analyze each stock in the basket
    all_results = []
    market_condition = None
    
    for symbol in symbols:
        print(f"\nAnalyzing {symbol}...")
        results = analyze_stock(symbol, analysis_date, stop_loss, take_profit)
        all_results.append(results)
        
        # Get market condition from first valid result
        if market_condition is None and "market_condition" in results:
            market_condition = results["market_condition"]
        
        # Print summary for this stock
        if "error" in results:
            print(f"Error analyzing {symbol}: {results['error']}")
            continue
            
        rec = results["recommendation"]["recommendation"]
        score = results["recommendation"]["score"]
        sentiment = results["sentiment_score"] if "sentiment_score" in results else "N/A"
        print(f"{symbol}: {rec} (Score: {score:.2f}, Sentiment: {sentiment:.4f})")

    # Allocate investment and calculate portfolio performance with market condition
    allocation = allocate_investment(total_investment, all_results, market_condition)
    portfolio = calculate_portfolio_performance(allocation, all_results, stop_loss, take_profit)
    
    # Print consolidated results for the basket
    print("\n===== BASKET ANALYSIS SUMMARY =====")
    print(f"Date: {analysis_date.strftime('%Y-%m-%d')}")
    print(f"Number of stocks analyzed: {len(all_results)}")
    print(f"Total investment: ${total_investment:.2f}")
    print(f"Risk Management: {stop_loss*100:.1f}% stop loss, {take_profit*100:.1f}% take profit")
    
    # Print recommendation based on market condition
    if market_condition:
        print(f"\nMarket Condition: {market_condition.upper()}")
        if market_condition == 'bearish':
            print("Recommendation: Reduce position sizes and be more selective with entries.")
        elif market_condition == 'volatile':
            print("Recommendation: Consider wider stop losses and reduced position sizes.")
    
    # If no stocks meet investment criteria, suggest holding cash
    if not allocation or sum(allocation.values()) == 0:
        print("\n\033[1mRECOMMENDATION: HOLD CASH\033[0m")
        print("No stocks meet the investment criteria. Consider waiting for better opportunities.")
        return
    
    # Count recommendations by type
    rec_counts = {}
    for result in all_results:
        if "recommendation" in result:
            rec = result["recommendation"]["recommendation"]
            rec_counts[rec] = rec_counts.get(rec, 0) + 1
    
    print("\nRecommendation breakdown:")
    for rec, count in rec_counts.items():
        print(f"- {rec}: {count} stocks ({count/len(all_results)*100:.1f}%)")
    
    # Calculate average sentiment
    valid_sentiments = [r.get("sentiment_score", 0) for r in all_results if "sentiment_score" in r]
    avg_sentiment = sum(valid_sentiments) / len(valid_sentiments) if valid_sentiments else 0
    print(f"\nAverage sentiment score: {avg_sentiment:.4f}")
    
    # Calculate combined win rate for the entire basket
    basket_win_rates = calculate_basket_win_rate(symbols)
    
    print("\n===== BASKET PERFORMANCE STATISTICS =====")
    print(f"Combined 3-Day Win Rate: {basket_win_rates['win_rate_3day']:.2f}% ({basket_win_rates.get('valid_trades_3day', 0)} trades)")
    print(f"Combined 5-Day Win Rate: {basket_win_rates['win_rate_5day']:.2f}% ({basket_win_rates.get('valid_trades_5day', 0)} trades)")
    print(f"Average 3-Day Return: {basket_win_rates.get('avg_return_3day', 0):.2f}%")
    print(f"Average 5-Day Return: {basket_win_rates.get('avg_return_5day', 0):.2f}%")
    print(f"Total historical trades analyzed: {basket_win_rates.get('total_trades', 0)}")
    
    # Display how many trades had stop loss or take profit triggered
    if 'stop_loss_count' in portfolio:
        print(f"Stop Loss Triggered: {portfolio['stop_loss_count']} stocks")
    if 'take_profit_count' in portfolio:
        print(f"Take Profit Triggered: {portfolio['take_profit_count']} stocks")
    
    # Display investment allocation and expected returns
    print("\n===== INVESTMENT ALLOCATION =====")
    for symbol, amount in allocation.items():
        if amount > 0:
            percent = amount / total_investment * 100
            print(f"{symbol}: ${amount:.2f} ({percent:.1f}%)")
    
    # Display projected portfolio returns in bold
    print("\n===== PROJECTED PORTFOLIO RETURNS =====")
    initial_investment = portfolio["initial_investment"]
    profit_loss_3day = portfolio["3day_value"] - initial_investment
    profit_loss_5day = portfolio["5day_value"] - initial_investment
    
    print(f"Initial investment: ${initial_investment:.2f}")
    print(f"3-Day projected return: {portfolio['3day_return_pct']:.2f}%")
    print(f"5-Day projected return: {portfolio['5day_return_pct']:.2f}%")
    
    # Bold formatting for profit/loss
    print(f"\n\033[1m3-Day Profit/Loss: ${profit_loss_3day:.2f}\033[0m")
    print(f"\033[1m5-Day Profit/Loss: ${profit_loss_5day:.2f}\033[0m")
    
    # Display detailed results for each stock
    print("\n===== DETAILED STOCK ANALYSIS =====")
    
    for result in all_results:
        symbol = result["symbol"]
        print(f"\n----- {symbol} -----")
        
        if "error" in result:
            print(f"Error: {result['error']}")
            continue
        
        # Display allocation for this stock
        if symbol in allocation:
            print(f"Investment: ${allocation[symbol]:.2f}")
            
        print(f"Recommendation: {result['recommendation']['recommendation']}")
        print(f"Score: {result['recommendation']['score']:.2f} (Technical: {result['recommendation']['technical_score']:.2f}, Sentiment: {result['sentiment_score']:.2f})")
        
        print("\nReasoning:")
        for reason in result["recommendation"]["reasoning"]:
            print(f"- {reason}")
        
        print("\nKey Technical Indicators:")
        tech = result["technical_indicators"]
        print(f"Price: ${tech['price']:.2f}")
        print(f"RSI (14): {tech['rsi']:.2f}")
        print(f"SMA (20/50/200): ${tech['sma_20']:.2f} / ${tech['sma_50']:.2f} / ${tech['sma_200']:.2f}")
        
        print("\nBacktest Results:")
        backtest = result["backtest"]
        print(f"3-Day Return: {backtest.get('day3_return', 'N/A'):.2f}% ({backtest.get('day3_direction', 'unknown')})")
        print(f"5-Day Return: {backtest.get('day5_return', 'N/A'):.2f}% ({backtest.get('day5_direction', 'unknown')})")
        
        # Show if stop loss or take profit was triggered
        if backtest.get('stop_loss_triggered', False):
            print(f"⚠️ Stop Loss triggered on {backtest.get('exit_date', 'N/A')} at ${backtest.get('exit_price', 'N/A'):.2f}")
        elif backtest.get('take_profit_triggered', False):
            print(f"✅ Take Profit triggered on {backtest.get('exit_date', 'N/A')} at ${backtest.get('exit_price', 'N/A'):.2f}")
        
        if "market_day3_return" in backtest:
            print(f"Market 3-Day Return: {backtest.get('market_day3_return', 'N/A'):.2f}%")
            print(f"Relative 3-Day Performance: {backtest.get('relative_day3_return', 'N/A'):.2f}%")

if __name__ == "__main__":
    main()
