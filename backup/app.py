from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import sqlite3
import random
import numpy as np
import traceback
import jinja2

from data_fetch import fetch_historical_data, fetch_market_data
from technical_analysis import calculate_indicators, analyze_indicators, calculate_rsi, calculate_bullish_engulfing, calculate_bearish_engulfing, calculate_doji, calculate_hammer, calculate_hanging_man, calculate_morning_star, calculate_evening_star, calculate_three_white_soldiers, calculate_three_black_crows, calculate_piercing_line, calculate_dark_cloud_cover
from backtesting import backtest_strategy
from sentiment_analysis import fetch_news_from_alpha_vantage, analyze_sentiment
from trends import get_trend, analyze_trend, stock_trends, save_trends

# Import database utilities
from database import get_db_connection, init_db

app = Flask(__name__)

# Add this at the top of your app.py file
def safe_round(value, precision=0):
    """Custom filter to safely round values that might be strings or non-numeric"""
    if value is None:
        return "N/A"
    
    try:
        # Try converting to float first
        float_value = float(value)
        # Check if it's NaN
        if pd.isna(float_value):
            return "N/A"
        return round(float_value, precision)
    except (ValueError, TypeError):
        # If conversion fails, return the original value
        return value

# Register the filter with Flask
app.jinja_env.filters['safe_round'] = safe_round

# Add this right after creating your Flask app
@app.template_filter('safe_round')
def safe_round_filter(value, precision=2):
    """
    A template filter that safely handles rounding of various value types.
    Returns 'N/A' for non-numeric values.
    """
    if value is None:
        return 'N/A'
    
    try:
        # Try to convert to float and round
        numeric_value = float(value)
        # Check for NaN
        if pd.isna(numeric_value):
            return 'N/A'
        return round(numeric_value, precision)
    except (ValueError, TypeError):
        # If conversion fails, return the original value
        return value

@app.route('/')
def index():
    """Dashboard showing portfolio overview and recent trades"""
    conn = get_db_connection()
    
    try:
        # Get portfolio data - convert to list of dictionaries to make them mutable
        portfolio = [dict(row) for row in conn.execute('SELECT * FROM portfolio').fetchall()]
        recent_trades = [dict(row) for row in conn.execute('SELECT * FROM trades ORDER BY date DESC LIMIT 10').fetchall()]
        
        # Calculate portfolio value and performance
        total_value = 0
        total_investment = 0
        daily_change = 0
        
        for position in portfolio:
            # Make sure necessary fields exist
            symbol = position.get('symbol', 'Unknown')
            shares = float(position.get('shares', 0))
            avg_price = float(position.get('avg_price', 0))
            
            # Mock current price (in real app, fetch from your data_fetch module)
            current_price = avg_price * 1.05  # Simulate 5% up for demo
            
            position_value = shares * current_price
            position_cost = shares * avg_price
            
            total_value += position_value
            total_investment += position_cost
            
            # Update the position dictionary with calculated fields
            position['current_price'] = current_price
            position['value'] = position_value
            position['profit_loss'] = position_value - position_cost
            position['percent_change'] = ((current_price / avg_price) - 1) * 100 if avg_price > 0 else 0
        
        total_pl = total_value - total_investment
        total_pl_percent = ((total_value / total_investment) - 1) * 100 if total_investment > 0 else 0
        
        # Calculate win rate from trades
        win_count = conn.execute('SELECT COUNT(*) FROM trades WHERE profit_loss > 0').fetchone()[0]
        total_count = conn.execute('SELECT COUNT(*) FROM trades').fetchone()[0]
        win_rate = (win_count / total_count * 100) if total_count > 0 else 0
        
        return render_template('index.html', 
                            portfolio=portfolio, 
                            recent_trades=recent_trades,
                            total_value=total_value,
                            total_pl=total_pl,
                            total_pl_percent=total_pl_percent,
                            win_rate=win_rate)
    finally:
        conn.close()

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        try:
            # Get form data
            symbols = request.form.get('symbols', '').split()
            investment = float(request.form.get('investment', 0))
            analysis_date_str = request.form.get('date', '')
            stop_loss = float(request.form.get('stop_loss', 5)) / 100
            take_profit = float(request.form.get('take_profit', 8)) / 100
            
            # Parse analysis date
            try:
                analysis_date = datetime.strptime(analysis_date_str, '%Y-%m-%d')
            except ValueError:
                # Default to a date we know has data
                analysis_date = datetime.strptime('2024-12-26', '%Y-%m-%d')
            
            # Process each symbol with fixed analysis function
            results = []
            for symbol in symbols:
                try:
                    print(f"Starting analysis for {symbol}...")
                    result = analyze_stock(symbol, analysis_date, stop_loss, take_profit)
                    print(f"Analysis for {symbol} completed")
                    results.append(result)
                except Exception as e:
                    print(f"Error analyzing {symbol}: {str(e)}")
                    results.append({
                        "symbol": symbol,
                        "error": str(e)
                    })
            
            print("All stocks analyzed, calculating allocation...")
            
            try:
                # Use improved allocation strategy with a try block
                allocation = allocate_investment(investment, results)
                print(f"Allocation calculated: {allocation}")
            except Exception as e:
                print(f"Error in allocation: {e}")
                # Fallback to equal allocation
                allocation = {}
                valid_count = len([r for r in results if "error" not in r])
                if valid_count > 0:
                    equal_amount = investment / valid_count
                    for result in results:
                        if "error" not in result:
                            allocation[result["symbol"]] = equal_amount
            
            # Update results with allocation
            for result in results:
                if "error" not in result:
                    symbol = result["symbol"]
                    result["allocation"] = allocation.get(symbol, 0.0)
            
            print("Calculating projections...")
            
            # Calculate projections with a try block
            try:
                projections = calculate_portfolio_projections(results)
            except Exception as e:
                print(f"Error calculating projections: {e}")
                projections = {
                    "total_investment": investment,
                    "3day_return_percent": 0,
                    "5day_return_percent": 0,
                    "3day_change": 0,
                    "5day_change": 0
                }
            
            print("Rendering template...")
            
            return render_template('analysis.html', 
                                  results=results, 
                                  investment=investment,
                                  projections=projections,
                                  analysis_date=analysis_date_str)
                                  
        except Exception as e:
            print(f"Error in analyze function: {str(e)}")
            import traceback
            print(traceback.format_exc())  # Print the full traceback
            return render_template('analysis.html', 
                                  error=f"An error occurred: {str(e)}",
                                  results=None)
    
    # For GET requests
    return render_template('analysis.html', results=None)

@app.route('/api/stock-data/<symbol>')
def stock_data(symbol):
    try:
        # Get start and end dates from query parameters or use defaults
        start_date = request.args.get('start_date', None)
        end_date = request.args.get('end_date', None)
        
        # Add debug logging
        print(f"Fetching stock data for {symbol} from {start_date} to {end_date}")
        
        # Convert string dates to datetime objects if provided
        if start_date:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            start_date = datetime.now() - timedelta(days=30)  # Default to last 30 days
            
        if end_date:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_date = datetime.now()
            
        # Fetch data with a wider date range to ensure we have enough for calculations
        historical_data = fetch_historical_data(symbol, start_date - timedelta(days=50), end_date)
        
        # Calculate technical indicators
        historical_data = calculate_indicators(historical_data)
        
        # Filter to requested date range after calculations
        filtered_data = historical_data[(historical_data.index >= pd.Timestamp(start_date)) & 
                                        (historical_data.index <= pd.Timestamp(end_date))]
        
        # Check if we have data
        if filtered_data.empty:
            return jsonify({'error': f'No data available for {symbol} in specified date range'})
        
        # Format the data for the chart
        chart_data = []
        for idx, row in filtered_data.iterrows():
            data_point = {
                'date': idx.strftime('%Y-%m-%d'),
                'price': round(float(row['Close']), 2),
                'open': round(float(row['Open']), 2),
                'high': round(float(row['High']), 2),
                'low': round(float(row['Low']), 2),
                'volume': int(row['Volume'])
            }
            
            # Add technical indicators if available
            if 'SMA_20' in row:
                data_point['sma20'] = round(float(row['SMA_20']), 2)
            if 'SMA_50' in row:
                data_point['sma50'] = round(float(row['SMA_50']), 2)
            if 'RSI' in row:
                data_point['rsi'] = round(float(row['RSI']), 2)
                
            chart_data.append(data_point)
        
        print(f"Returning {len(chart_data)} data points for {symbol}")
        return jsonify(chart_data)
    
    except Exception as e:
        print(f"Error in stock_data API for {symbol}: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/execute-trade', methods=['POST'])
def execute_trade():
    """Execute and record a trade with missed opportunity tracking"""
    data = request.json
    
    # Calculate current profit/loss
    current_profit_loss = 0
    current_profit_loss_percent = 0
    
    # Save trade to database
    conn = get_db_connection()
    
    try:
        if data['action'] == 'SELL':
            # Get average purchase price and entry date
            portfolio_row = conn.execute(
                'SELECT symbol, avg_price, shares, entry_date FROM portfolio WHERE symbol = ?', 
                (data['symbol'],)
            ).fetchone()
            
            if portfolio_row:
                avg_price = portfolio_row['avg_price']
                current_shares = portfolio_row['shares']
                entry_date = portfolio_row['entry_date']
                
                if data['shares'] > current_shares:
                    return jsonify({"status": "error", "message": f"Cannot sell more shares than you own. You have {current_shares} shares."})
                    
                # Calculate current profit/loss
                current_profit_loss = (data['price'] - avg_price) * data['shares']
                current_profit_loss_percent = ((data['price'] / avg_price) - 1) * 100
                
                # Calculate potential missed opportunity (30 days forward)
                # Fetch future data to calculate missed opportunity
                sell_date = datetime.strptime(data['date'], '%Y-%m-%d')
                future_date = sell_date + timedelta(days=30)
                
                try:
                    future_data = fetch_historical_data(data['symbol'], sell_date, future_date)
                    
                    if not future_data.empty:
                        # Find the maximum price in the next 30 days
                        max_future_price = future_data['High'].max()
                        
                        # Calculate potential missed profit
                        potential_max_profit = (max_future_price - avg_price) * data['shares']
                        potential_max_profit_percent = ((max_future_price / avg_price) - 1) * 100
                        
                        # Calculate opportunity cost
                        missed_opportunity = potential_max_profit - current_profit_loss
                        missed_opportunity_percent = potential_max_profit_percent - current_profit_loss_percent
                        
                        # Store missed opportunity information
                        conn.execute(
                            'INSERT INTO missed_opportunities (trade_id, max_potential_price, max_potential_profit, max_potential_profit_percent, missed_opportunity, missed_opportunity_percent, max_price_date) VALUES (?, ?, ?, ?, ?, ?, ?)',
                            ((conn.lastrowid or 0) + 1, max_future_price, potential_max_profit, potential_max_profit_percent, missed_opportunity, missed_opportunity_percent, future_data['High'].idxmax().strftime('%Y-%m-%d'))
                        )
                except Exception as e:
                    print(f"Error calculating missed opportunity: {e}")
        
        # Insert trade record with profit/loss info
        trade_id = conn.execute(
            'INSERT INTO trades (symbol, date, price, shares, action, investment, stop_loss, take_profit, profit_loss, profit_loss_percent, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (data['symbol'], data['date'], data['price'], data['shares'], data['action'], 
             data['investment'], data['stop_loss'], data['take_profit'], current_profit_loss, current_profit_loss_percent, data.get('notes', ''))
        ).lastrowid
        
        # Update portfolio
        if data['action'] == 'BUY':
            # Add entry_date for tracking when position was opened
            existing = conn.execute('SELECT * FROM portfolio WHERE symbol = ?', (data['symbol'],)).fetchone()
            
            if existing:
                # Update existing position (average down/up)
                existing_shares = existing['shares']
                existing_avg_price = existing['avg_price']
                
                new_shares = existing_shares + data['shares']
                new_avg_price = ((existing_shares * existing_avg_price) + (data['shares'] * data['price'])) / new_shares
                
                conn.execute(
                    'UPDATE portfolio SET shares = ?, avg_price = ?, last_updated = ? WHERE symbol = ?',
                    (new_shares, new_avg_price, datetime.now().strftime('%Y-%m-%d'), data['symbol'])
                )
            else:
                # Add new position with entry date
                conn.execute(
                    'INSERT INTO portfolio (symbol, shares, avg_price, entry_date, last_updated) VALUES (?, ?, ?, ?, ?)',
                    (data['symbol'], data['shares'], data['price'], data['date'], datetime.now().strftime('%Y-%m-%d'))
                )
        elif data['action'] == 'SELL':
            # Update remaining shares
            existing = conn.execute('SELECT * FROM portfolio WHERE symbol = ?', (data['symbol'],)).fetchone()
            
            if existing:
                remaining_shares = existing['shares'] - data['shares']
                
                if remaining_shares <= 0:
                    # Remove from portfolio if no shares left
                    conn.execute('DELETE FROM portfolio WHERE symbol = ?', (data['symbol'],))
                else:
                    # Update with remaining shares (keep same avg price)
                    conn.execute(
                        'UPDATE portfolio SET shares = ?, last_updated = ? WHERE symbol = ?',
                        (remaining_shares, datetime.now().strftime('%Y-%m-%d'), data['symbol'])
                    )
        
        conn.commit()
        
        # Prepare response with missed opportunity data if applicable
        response_data = {
            "status": "success", 
            "message": "Trade executed successfully", 
            "profit_loss": current_profit_loss,
            "profit_loss_percent": current_profit_loss_percent,
        }
        
        # Add missed opportunity data if it's a sell
        if data['action'] == 'SELL':
            missed_opp = conn.execute('SELECT * FROM missed_opportunities WHERE trade_id = ?', (trade_id,)).fetchone()
            if missed_opp:
                response_data["missed_opportunity"] = {
                    "max_potential_profit_percent": missed_opp['max_potential_profit_percent'],
                    "missed_opportunity_percent": missed_opp['missed_opportunity_percent'],
                    "max_price_date": missed_opp['max_price_date']
                }
        
        return jsonify(response_data)
    
    except Exception as e:
        conn.rollback()
        return jsonify({"status": "error", "message": str(e)})
    finally:
        conn.close()

@app.route('/portfolio')
def portfolio():
    """View and manage portfolio"""
    conn = get_db_connection()
    
    try:
        # Get portfolio data
        portfolio = conn.execute('SELECT * FROM portfolio').fetchall()
        
        # Get trades with missed opportunities data using a JOIN
        trades = conn.execute('''
            SELECT 
                t.*, 
                mo.missed_opportunity_percent, 
                mo.max_price_date 
            FROM 
                trades t 
            LEFT JOIN 
                missed_opportunities mo ON t.id = mo.trade_id 
            ORDER BY 
                t.date DESC
        ''').fetchall()
        
        # Calculate performance for each position
        for position in portfolio:
            position = dict(position)
            # Mock current price (in real app, fetch from API)
            current_price = position['avg_price'] * 1.05  # 5% up for demo
            position['current_price'] = current_price
            position['profit_loss'] = (current_price - position['avg_price']) * position['shares']
            position['percent_change'] = (current_price - position['avg_price']) / position['avg_price'] * 100
        
        return render_template('portfolio.html', portfolio=portfolio, trades=trades)
    except Exception as e:
        print(f"Error in portfolio route: {e}")
        return render_template('portfolio.html', portfolio=[], trades=[], error=str(e))
    finally:
        conn.close()

def analyze_stock(symbol, analysis_date, stop_loss=0.05, take_profit=0.08):
    results = {"symbol": symbol, "date": analysis_date.strftime('%Y-%m-%d')}
    
    try:
        print(f"Analyzing {symbol} for date: {analysis_date}")
        
        # Fetch historical data
        historical_data = fetch_historical_data(symbol, analysis_date - timedelta(days=60), analysis_date + timedelta(days=30))
        
        # Calculate indicators
        historical_data = calculate_indicators(historical_data)
        
        # Calculate chart patterns
        historical_data = calculate_chart_patterns(historical_data)
        
        # Find the closest date to analysis_date
        if pd.Timestamp(analysis_date) in historical_data.index:
            closest_date = pd.Timestamp(analysis_date)
        else:
            available_dates = historical_data.index.tolist()
            closest_date = min(available_dates, key=lambda x: abs(x - pd.Timestamp(analysis_date)))
            print(f"Using closest date {closest_date} for {symbol}")
        
        # Get single row values
        price = historical_data.loc[closest_date, 'Close']
        results["price_on_date"] = price
        
        # Technical indicators
        results["technical_indicators"] = {}
        
        # Extract single values for indicators
        try:
            if 'RSI' in historical_data.columns:
                results["technical_indicators"]["rsi"] = float(historical_data.loc[closest_date, 'RSI'])
                
            if 'MACD' in historical_data.columns:
                results["technical_indicators"]["macd"] = float(historical_data.loc[closest_date, 'MACD'])
                
            if 'MACD_Signal' in historical_data.columns:
                results["technical_indicators"]["macd_signal"] = float(historical_data.loc[closest_date, 'MACD_Signal'])
                
            if 'SMA_20' in historical_data.columns:
                results["technical_indicators"]["sma_20"] = float(historical_data.loc[closest_date, 'SMA_20'])
                
            if 'SMA_50' in historical_data.columns:
                results["technical_indicators"]["sma_50"] = float(historical_data.loc[closest_date, 'SMA_50'])
        except (KeyError, ValueError) as e:
            print(f"Error extracting indicators for {symbol}: {e}")
        
        # Generate signals dictionary with proper scalar values
        signals = {}
        signals['trend'] = {}
        signals['momentum'] = {}
        signals['patterns'] = {}  # Pattern signals
        
        # Check if we have the necessary indicators
        if "technical_indicators" in results:
            tech = results["technical_indicators"]
            
            # Trend signals
            if "price_on_date" in results and "sma_20" in tech and "sma_50" in tech:
                price_value = float(results["price_on_date"])
                sma20_value = float(tech["sma_20"])
                sma50_value = float(tech["sma_50"])
                
                # Compare scalar values, not Series
                signals['trend']['price_above_sma_20'] = price_value > sma20_value
                signals['trend']['price_above_sma_50'] = price_value > sma50_value
                signals['trend']['sma_20_above_50'] = sma20_value > sma50_value
            
            # Momentum signals
            if "rsi" in tech:
                rsi_value = float(tech["rsi"])
                signals['momentum']['rsi_oversold'] = rsi_value < 30
                signals['momentum']['rsi_overbought'] = rsi_value > 70
            
            if "macd" in tech and "macd_signal" in tech:
                macd_value = float(tech["macd"])
                macd_signal_value = float(tech["macd_signal"])
                signals['momentum']['macd_above_signal'] = macd_value > macd_signal_value
        
        # Add chart pattern signals
        pattern_columns = [col for col in historical_data.columns if col.startswith('pattern_')]
        for col in pattern_columns:
            pattern_name = col.replace('pattern_', '')
            signals['patterns'][pattern_name] = bool(historical_data.loc[closest_date, col])
        
        # Generate sentiment score (THIS WAS MISSING)
        try:
            news_articles = fetch_news_from_alpha_vantage(symbol)
            sentiment_score = analyze_sentiment(news_articles)
            results["sentiment_score"] = sentiment_score
            print(f"Sentiment score for {symbol}: {sentiment_score}")
        except Exception as e:
            print(f"Error fetching sentiment for {symbol}: {e}")
            sentiment_score = 0.5  # Neutral default
            results["sentiment_score"] = sentiment_score
        
        # Get market condition
        market_condition = analyze_market_condition()
        results["market_condition"] = market_condition
        
        # Generate recommendation using our enhanced function
        recommendation = generate_recommendation(signals, sentiment_score)
        results["recommendation"] = recommendation
        
        # Backtesting
        backtest_results = backtest_strategy(historical_data, closest_date, None, stop_loss, take_profit)
        results["backtest"] = backtest_results
        
        # Calculate missed opportunity (future potential)
        try:
            future_data = historical_data[historical_data.index > closest_date]
            
            if not future_data.empty:
                # Find the maximum price in the future data
                max_future_price = future_data['High'].max()
                max_price_date = future_data['High'].idxmax().strftime('%Y-%m-%d')
                
                # Calculate potential maximum profit percent
                max_potential_profit_percent = ((max_future_price / price) - 1) * 100
                
                # Calculate missed opportunity (difference between max potential and actual)
                actual_return = backtest_results.get("day5_return", 0)
                if actual_return is None:
                    actual_return = 0
                
                missed_opportunity_percent = max_potential_profit_percent - actual_return
                
                # Add to backtest results
                backtest_results["max_potential_price"] = max_future_price
                backtest_results["max_price_date"] = max_price_date
                backtest_results["max_potential_profit_percent"] = max_potential_profit_percent
                backtest_results["missed_opportunity_percent"] = missed_opportunity_percent
                
                print(f"Missed opportunity for {symbol}: {missed_opportunity_percent:.2f}%")
        except Exception as e:
            print(f"Error calculating missed opportunity for {symbol}: {e}")
        
        # Calculate entry timing
        entry_timing = analyze_entry_timing(symbol, historical_data)
        results["entry_timing"] = entry_timing
        
        return results
        
    except Exception as e:
        import traceback
        print(f"Error analyzing {symbol}: {e}")
        print(traceback.format_exc())
        results["error"] = str(e)
        return results

@app.route('/stock/<symbol>')
def stock_detail(symbol):
    """Display detailed view of a stock"""
    try:
        # Clear caches for this individual stock analysis
        clear_sentiment_cache()
        clear_trend_cache()
        
        # Get latest stock data
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Create a stock object with the necessary information
        stock = {
            'symbol': symbol,
            'name': info.get('shortName', symbol),
            'price': info.get('currentPrice', 0),
            'previousClose': info.get('previousClose', 0),
            'change': info.get('currentPrice', 0) - info.get('previousClose', 0),
            'changePercent': ((info.get('currentPrice', 0) - info.get('previousClose', 0)) / info.get('previousClose', 1)) * 100,
            'volume': info.get('volume', 0),
            'marketCap': info.get('marketCap', 0),
            'recommendation': 'STRONG BUY'  # This would come from your analysis logic
        }
        
        return render_template('stock_details.html', stock=stock)
    except Exception as e:
        # Instead of using flash, redirect with message parameter
        message = f"Error loading stock details: {str(e)}"
        return redirect(url_for('index', message=message, type='danger'))

@app.route('/debug/chart-data/<symbol>')
def debug_chart_data(symbol):
    """Debug endpoint to test chart data"""
    days = int(request.args.get('days', 30))
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        # Use yfinance to get data
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        # Basic HTML representation
        html = f"<h1>Debug Chart Data for {symbol}</h1>"
        html += f"<p>Fetched {len(df)} data points</p>"
        
        if not df.empty:
            html += "<table border='1'><tr><th>Date</th><th>Open</th><th>Close</th><th>SMA20</th></tr>"
            
            # Calculate SMA20
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            
            for date, row in df.iterrows():
                html += f"<tr><td>{date.strftime('%Y-%m-%d')}</td><td>${row['Open']:.2f}</td><td>${row['Close']:.2f}</td>"
                html += f"<td>${row['SMA_20']:.2f if not pd.isna(row['SMA_20']) else 'N/A'}</td></tr>"
            
            html += "</table>"
        else:
            html += "<p>No data returned</p>"
            
        return html
    except Exception as e:
        return f"<h1>Error</h1><p>{str(e)}</p><pre>{traceback.format_exc()}</pre>"

@app.route('/debug/rsi/<symbol>')
def debug_rsi(symbol):
    """Debug endpoint to test RSI calculation"""
    try:
        import yfinance as yf
        from technical_analysis import calculate_rsi
        
        # Get data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="3mo")
        
        # Calculate RSI
        df['RSI'] = calculate_rsi(df['Close'])
        
        # Create simple HTML table
        html = f"<h1>RSI Calculation Debug for {symbol}</h1>"
        html += f"<p>Successfully calculated RSI for {len(df)} data points</p>"
        
        html += "<table border='1'><tr><th>Date</th><th>Close</th><th>RSI</th></tr>"
        for date, row in df.tail(20).iterrows():
            html += f"<tr><td>{date.strftime('%Y-%m-%d')}</td><td>${row['Close']:.2f}</td>"
            html += f"<td>{row['RSI']:.2f if not pd.isna(row['RSI']) else 'N/A'}</td></tr>"
        html += "</table>"
        
        return html
    except Exception as e:
        import traceback
        return f"<h1>Error Testing RSI</h1><p>{str(e)}</p><pre>{traceback.format_exc()}</pre>"

# Add this function to your app.py to ensure backtest values are always numeric
def ensure_numeric_backtest_results(backtest_results):
    """Ensure all backtest results fields have numeric values"""
    # Make sure necessary fields exist and are numeric
    if backtest_results.get('day3_return') is None:
        backtest_results['day3_return'] = 0.0
    else:
        try:
            backtest_results['day3_return'] = float(backtest_results['day3_return'])
        except (TypeError, ValueError):
            backtest_results['day3_return'] = 0.0
            
    if backtest_results.get('day5_return') is None:
        backtest_results['day5_return'] = 0.0
    else:
        try:
            backtest_results['day5_return'] = float(backtest_results['day5_return'])
        except (TypeError, ValueError):
            backtest_results['day5_return'] = 0.0
    
    return backtest_results

def calculate_actual_returns(df, analysis_date):
    """
    Calculate ACTUAL returns 3 and 5 trading days after the analysis date
    
    :param df: DataFrame with historical price data
    :param analysis_date: The date used for analysis (string or datetime)
    :return: Dictionary with actual 3-day and 5-day returns
    """
    print(f"Calculating actual returns after {analysis_date}")
    
    # Convert to pandas timestamp if it's a string
    if isinstance(analysis_date, str):
        try:
            analysis_date = pd.Timestamp(analysis_date)
        except ValueError:
            print(f"Error parsing date: {analysis_date}")
            return {'error': f"Invalid date format: {analysis_date}"}
    
    # Ensure the analysis_date is a pandas Timestamp
    if not isinstance(analysis_date, pd.Timestamp):
        analysis_date = pd.Timestamp(analysis_date)
    
    print(f"Using analysis date: {analysis_date.strftime('%Y-%m-%d')}")
    
    # Find the index of the analysis date or the closest trading day before it
    try:
        # Get available dates in the dataframe
        available_dates = df.index
        
        # Find closest date to analysis_date that exists in our data
        # First check if analysis_date is in our data
        if analysis_date in available_dates:
            start_idx = available_dates.get_loc(analysis_date)
        else:
            # Get all dates at or before analysis_date
            valid_dates = available_dates[available_dates <= analysis_date]
            if len(valid_dates) == 0:
                return {'error': f"No data available before {analysis_date.strftime('%Y-%m-%d')}"}
            
            # Get the closest previous date
            closest_date = valid_dates[-1]
            start_idx = available_dates.get_loc(closest_date)
            print(f"Analysis date {analysis_date.strftime('%Y-%m-%d')} not found, using closest date: {closest_date.strftime('%Y-%m-%d')}")
        
        # Get the date used as starting point
        start_date = available_dates[start_idx]
        start_price = df.loc[start_date, 'Close']
        
        print(f"Start date: {start_date.strftime('%Y-%m-%d')}, price: ${start_price:.2f}")
        
        # Find indices for 3 and 5 trading days later
        day3_idx = min(start_idx + 3, len(available_dates) - 1)
        day5_idx = min(start_idx + 5, len(available_dates) - 1)
        
        # Get the actual dates and prices
        day3_date = available_dates[day3_idx]
        day5_date = available_dates[day5_idx]
        
        day3_price = df.loc[day3_date, 'Close']
        day5_price = df.loc[day5_date, 'Close']
        
        print(f"3-day date: {day3_date.strftime('%Y-%m-%d')}, price: ${day3_price:.2f}")
        print(f"5-day date: {day5_date.strftime('%Y-%m-%d')}, price: ${day5_price:.2f}")
        
        # Calculate percentage returns
        day3_return = ((day3_price - start_price) / start_price) * 100
        day5_return = ((day5_price - start_price) / start_price) * 100
        
        print(f"3-day return: {day3_return:.2f}%, 5-day return: {day5_return:.2f}%")
        
        return {
            'day3_return': round(day3_return, 2),
            'day5_return': round(day5_return, 2),
            'day3_date': day3_date.strftime('%Y-%m-%d'),
            'day5_date': day5_date.strftime('%Y-%m-%d'),
            'day3_direction': 'up' if day3_return > 0 else 'down',
            'day5_direction': 'up' if day5_return > 0 else 'down'
        }
    except Exception as e:
        import traceback
        print(f"Error calculating actual returns: {str(e)}")
        print(traceback.format_exc())
        return {'error': str(e)}

# Add this function to clear trend data cache
def clear_trend_cache():
    """
    Clear the trend data cache to ensure fresh trend data for each analysis.
    """
    global stock_trends
    stock_trends.clear()
    # Optionally remove the JSON file too
    if os.path.exists('stock_trends.json'):
        try:
            os.remove('stock_trends.json')
            print("Removed stock_trends.json file")
        except Exception as e:
            print(f"Could not remove trends file: {e}")
    
    print("Trend data cache cleared")

# Add this to your sentiment_analysis.py file or create a wrapper function in app.py
news_cache = {}  # Add this global variable to app.py

def clear_sentiment_cache():
    """
    Clear the sentiment data cache to ensure fresh sentiment data for each analysis.
    """
    global news_cache
    news_cache.clear()
    print("Sentiment cache cleared")

@app.route('/debug/price/<symbol>/<date>')
def debug_price(symbol, date):
    """Debug endpoint to check price data for a specific symbol and date."""
    try:
        analysis_date = datetime.strptime(date, '%Y-%m-%d')
        
        # Fetch fresh data with no caching
        data = fetch_historical_data(symbol, analysis_date, force_refresh=True)
        
        # Find closest available date
        if pd.Timestamp(analysis_date) in data.index:
            closest_date = pd.Timestamp(analysis_date)
        else:
            available_dates = data.index.tolist()
            closest_date = min(available_dates, key=lambda x: abs(x - pd.Timestamp(analysis_date)))
        
        price = data.loc[closest_date, 'Close']
        
        # Get multiple dates around the target for verification
        date_range = []
        for i in range(-5, 6):
            check_date = closest_date + pd.Timedelta(days=i)
            if check_date in data.index:
                date_range.append({
                    'date': check_date.strftime('%Y-%m-%d'),
                    'price': float(data.loc[check_date, 'Close']),
                    'is_target': i == 0
                })
        
        return jsonify({
            'symbol': symbol,
            'requested_date': date,
            'closest_date': closest_date.strftime('%Y-%m-%d'),
            'price': float(price),
            'date_range': date_range,
            'data_source': 'Alpha Vantage API with force refresh'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'trace': traceback.format_exc()
        })

def can_be_rounded(value):
    """Check if a value can be safely rounded"""
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    return False

def preprocess_results(results):
    """
    Preprocess results to make sure all values are safely roundable.
    """
    if not results:
        return results
        
    processed_results = []
    
    for result in results:
        # Start with a copy of the original result
        processed = result.copy()
        
        # Process backtest data
        if "backtest" in processed and processed["backtest"]:
            for key in processed["backtest"]:
                if key.endswith("_return") or key.endswith("_price"):
                    try:
                        if processed["backtest"][key] is not None:
                            processed["backtest"][key] = float(processed["backtest"][key])
                    except (ValueError, TypeError):
                        processed["backtest"][key] = "N/A"
        
        # Process technical indicators
        if "technical_indicators" in processed and processed["technical_indicators"]:
            for key in processed["technical_indicators"]:
                try:
                    if processed["technical_indicators"][key] is not None:
                        processed["technical_indicators"][key] = float(processed["technical_indicators"][key])
                except (ValueError, TypeError):
                    processed["technical_indicators"][key] = "N/A"
        
        # Process other common numeric fields
        for key in ["price_on_date", "allocation"]:
            if key in processed:
                try:
                    if processed[key] is not None:
                        processed[key] = float(processed[key])
                except (ValueError, TypeError):
                    processed[key] = 0.0
        
        processed_results.append(processed)
    
    return processed_results

def allocate_investment(total_amount, results, market_condition=None):
    """
    Allocate investment amount ONLY to stocks with BUY recommendations AND good entry timing.
    
    :param total_amount: Total investment amount
    :param results: Analysis results for multiple stocks
    :param market_condition: Current market condition
    :return: Dictionary of allocations by symbol
    """
    allocations = {}
    buy_candidates = []
    
    # Risk adjustment based on market condition
    risk_multiplier = 1.0
    if market_condition == 'bearish':
        risk_multiplier = 0.7
    elif market_condition == 'volatile':
        risk_multiplier = 0.8
    
    # Adjust total amount based on market condition
    adjusted_total = total_amount * risk_multiplier
    
    # First pass: collect ONLY stocks with both BUY recommendations AND good entry timing
    for result in results:
        if 'error' not in result and 'recommendation' in result:
            rec = result.get('recommendation', {}).get('recommendation', '')
            if 'BUY' in rec:
                # Check entry timing - ONLY include if good entry
                entry_quality = result.get('entry_timing', {}).get('good_entry', False)
                
                # STRICT REQUIREMENT: Must have good entry timing
                if entry_quality:
                    score = result.get('recommendation', {}).get('score', 0)
                    signal_strength = result.get('recommendation', {}).get('signal_strength', 0.5)
                    buy_candidates.append((result, signal_strength))
    
    # If no buy candidates found with good entry timing, return empty allocations
    if not buy_candidates:
        print("No BUY recommendations with good entry timing found")
        return allocations
    
    # Calculate total score for weighting
    total_weight = sum(signal_strength for _, signal_strength in buy_candidates)
    
    if total_weight == 0:
        # Fallback to equal weighting if all weights are zero
        per_stock_amount = adjusted_total / len(buy_candidates)
        for result, _ in buy_candidates:
            symbol = result['symbol']
            allocations[symbol] = per_stock_amount
            print(f"Equal allocation: ${per_stock_amount:.2f} to {symbol}")
    else:
        # Weighted allocation based on signal strength
        for result, signal_strength in buy_candidates:
            symbol = result['symbol']
            weight = signal_strength / total_weight
            allocation = adjusted_total * weight
            
            # Ensure minimum and maximum allocation
            allocation = max(adjusted_total * 0.05, min(adjusted_total * 0.4, allocation))
            
            allocations[symbol] = allocation
            print(f"Weighted allocation: ${allocation:.2f} to {symbol} (weight: {weight:.2f})")
    
    return allocations

def calculate_portfolio_projections(results):
    """
    Calculate projected portfolio returns based on actual 3-day and 5-day returns.
    Now includes missed opportunity calculations.
    """
    total_investment = 0
    total_3day_value = 0
    total_5day_value = 0
    total_missed_opportunity = 0
    total_missed_opportunity_percent = 0
    
    # Track how many stocks have missed opportunity data
    stocks_with_missed_opp = 0
    
    for result in results:
        # Skip errors
        if "error" in result:
            continue
            
        # Get allocation - use dictionary key access
        allocation = result.get("allocation", 0)
        if allocation is None:
            allocation = 0
        
        # Convert to float if it's a string
        if isinstance(allocation, str):
            try:
                allocation = float(allocation.replace('$', '').replace(',', ''))
            except (ValueError, AttributeError):
                allocation = 0
        
        # Get 3-day and 5-day returns
        backtest = result.get("backtest", {})
        if not backtest:
            continue
            
        # Process missed opportunity if available
        if 'missed_opportunity_percent' in backtest:
            missed_opp_pct = backtest.get('missed_opportunity_percent', 0)
            # Calculate missed opportunity in dollars based on allocation
            missed_opp = allocation * missed_opp_pct / 100
            total_missed_opportunity += missed_opp
            stocks_with_missed_opp += 1
            
        # Regular projection calculations
        day3_return = backtest.get("day3_return", 0)
        if day3_return is None or (isinstance(day3_return, str) and day3_return.lower() == 'n/a'):
            day3_return = 0
        
        # Convert to float if it's a string
        if isinstance(day3_return, str):
            try:
                day3_return = float(day3_return.replace('%', ''))
            except (ValueError, AttributeError):
                day3_return = 0
        
        # Get 5-day return
        day5_return = backtest.get("day5_return", 0)
        if day5_return is None or (isinstance(day5_return, str) and day5_return.lower() == 'n/a'):
            day5_return = 0
        
        # Convert to float if it's a string
        if isinstance(day5_return, str):
            try:
                day5_return = float(day5_return.replace('%', ''))
            except (ValueError, AttributeError):
                day5_return = 0
        
        # Calculate projected values
        day3_value = allocation * (1 + day3_return / 100)
        day5_value = allocation * (1 + day5_return / 100)
        
        # Add to totals
        total_investment += allocation
        total_3day_value += day3_value
        total_5day_value += day5_value
    
    # Calculate return percentages
    return3_day = ((total_3day_value - total_investment) / total_investment * 100) if total_investment > 0 else 0
    return5_day = ((total_5day_value - total_investment) / total_investment * 100) if total_investment > 0 else 0
    
    # Calculate missed opportunity percentage
    if total_investment > 0 and stocks_with_missed_opp > 0:
        total_missed_opportunity_percent = (total_missed_opportunity / total_investment) * 100
    
    # Create result dictionary
    projections = {
        "total_investment": total_investment,
        "3day_value": total_3day_value,
        "5day_value": total_5day_value,
        "3day_return_percent": return3_day,
        "5day_return_percent": return5_day,
        "3day_change": total_3day_value - total_investment,
        "5day_change": total_5day_value - total_investment,
        "total_missed_opportunity": total_missed_opportunity,
        "total_missed_opportunity_percent": total_missed_opportunity_percent
    }
    
    return projections

def analyze_market_condition():
    """
    Analyze overall market condition to determine if it's safe to invest.
    Returns: 'bullish', 'bearish', or 'neutral'
    """
    try:
        # Get SPY (S&P 500 ETF) data
        spy_data = fetch_historical_data('SPY', datetime.now() - timedelta(days=60), datetime.now())
        spy_data = calculate_indicators(spy_data)
        
        # Check market trend
        is_above_sma50 = spy_data['Close'].iloc[-1] > spy_data['SMA_50'].iloc[-1]
        is_above_sma200 = spy_data['Close'].iloc[-1] > spy_data['SMA_200'].iloc[-1]
        
        # Check momentum
        rsi = spy_data['RSI'].iloc[-1]
        macd = spy_data['MACD'].iloc[-1]
        macd_signal = spy_data['MACD_Signal'].iloc[-1]
        
        # Determine market condition
        if is_above_sma50 and is_above_sma200 and macd > macd_signal and rsi > 50:
            return 'bullish'
        elif not is_above_sma50 and not is_above_sma200 and macd < macd_signal and rsi < 50:
            return 'bearish'
        else:
            return 'neutral'
    except Exception as e:
        print(f"Error analyzing market condition: {e}")
        return 'neutral'  # Default to neutral if error

def analyze_entry_timing(symbol, historical_data):
    """
    Analyze if the current stock market open is a good entry point.
    Focuses specifically on market open conditions.
    
    :param symbol: Stock symbol
    :param historical_data: DataFrame with historical data
    :return: Dictionary with entry timing information
    """
    result = {'good_entry': False, 'reasons': []}
    
    if historical_data.empty or len(historical_data) < 5:
        result['reasons'].append("Insufficient historical data")
        return result
    
    # Get the most recent day (today)
    today = historical_data.iloc[-1]
    
    # Calculate gap (difference between today's open and yesterday's close)
    if len(historical_data) >= 2:
        yesterday = historical_data.iloc[-2]
        yesterday_close = yesterday['Close']
        today_open = today['Open']
        
        # Calculate gap percentage
        gap_percent = (today_open - yesterday_close) / yesterday_close * 100
        
        # Small gap down: good entry at market open (-0.5% to -2%)
        if -2.0 <= gap_percent <= -0.5:
            result['good_entry'] = True
            result['reasons'].append(f"Small gap down at market open ({gap_percent:.2f}%)")
        
        # Gap up with strong volume: good momentum entry
        if gap_percent > 0.5 and today['Volume'] > yesterday['Volume'] * 1.2:
            result['good_entry'] = True
            result['reasons'].append(f"Gap up with strong volume at market open ({gap_percent:.2f}%)")
    
    # First 15-minute price action (using HIGH-LOW range)
    # Note: For simulation purposes only - real implementation would need intraday data
    day_range = today['High'] - today['Low']
    day_avg_range = historical_data['High'].iloc[-5:].mean() - historical_data['Low'].iloc[-5:].mean()
    
    # Low volatility open (tight range): good for controlled entries
    if day_range < day_avg_range * 0.7:
        result['good_entry'] = True
        result['reasons'].append("Low volatility at market open")
    
    # Check pre-market patterns that affect market open
    # (Simulated - would need actual pre-market data)
    if 'RSI' in today and today['RSI'] < 30 and today['Open'] > today['Low']:
        result['good_entry'] = True
        result['reasons'].append(f"Oversold condition with rising price from open (RSI: {today['RSI']:.1f})")
    
    # Market open near key support level
    if 'SMA_20' in today and 0.99 <= today['Open']/today['SMA_20'] <= 1.01:
        result['good_entry'] = True
        result['reasons'].append("Opening near key support level (20-day MA)")
        
    # Opening below VWAP (simulated - would need actual VWAP)
    if 'Close' in today and today['Open'] < today['Close'] * 0.995:
        result['good_entry'] = True
        result['reasons'].append("Opening below VWAP, potential for mean reversion")
    
    return result

# Define sectors for common stocks
SECTORS = {
    'AAPL': 'Technology',
    'MSFT': 'Technology',
    'AMZN': 'Consumer Cyclical',
    'GOOGL': 'Communication Services',
    'META': 'Communication Services',
    'TSLA': 'Consumer Cyclical',
    'NVDA': 'Technology',
    'JPM': 'Financial Services',
    'BAC': 'Financial Services',
    'WMT': 'Consumer Defensive',
    'JNJ': 'Healthcare',
    'PFE': 'Healthcare',
    'XOM': 'Energy',
    'CVX': 'Energy',
    # Add more as needed
}

def ensure_diversification(allocation, results, min_sectors=2):
    """Ensure portfolio is diversified across sectors"""
    if not allocation:
        return allocation
    
    # Map symbols to sectors
    sectors = {}
    for result in results:
        symbol = result["symbol"]
        if symbol in SECTORS:
            sectors[symbol] = SECTORS[symbol]
        else:
            sectors[symbol] = 'Unknown'
    
    # Count sectors in current allocation
    allocated_sectors = set(sectors[symbol] for symbol in allocation if symbol in sectors)
    
    # If already diversified, return original allocation
    if len(allocated_sectors) >= min_sectors:
        return allocation
    
    # Otherwise, try to add stocks from missing sectors
    all_sectors = set(sectors.values())
    missing_sectors = all_sectors - allocated_sectors
    
    # Find stocks from missing sectors with best scores
    new_allocation = allocation.copy()
    for result in results:
        symbol = result["symbol"]
        if symbol in sectors and sectors[symbol] in missing_sectors and symbol not in new_allocation:
            # Add a small allocation to this stock for diversification
            new_allocation[symbol] = min(allocation.values()) if allocation else 0
            
            # Stop when we've added enough sectors
            if len(set(sectors[s] for s in new_allocation if s in sectors)) >= min_sectors:
                break
    
    # Rebalance to maintain total investment
    original_total = sum(allocation.values())
    new_total = sum(new_allocation.values())
    
    if new_total > 0:
        scaling_factor = original_total / new_total
        for symbol in new_allocation:
            new_allocation[symbol] *= scaling_factor
    
    return new_allocation

def optimize_risk_parameters(symbol, historical_data, default_stop_loss=0.05, default_take_profit=0.08):
    """
    Optimize stop loss and take profit based on stock's volatility
    """
    try:
        # Calculate volatility (standard deviation of returns)
        returns = historical_data['Close'].pct_change().dropna()
        volatility = returns.std()
        
        # Scale stop loss and take profit based on volatility
        volatility_multiplier = 1.0
        if volatility > 0.03:  # High volatility
            volatility_multiplier = 1.5  # Wider stop loss for volatile stocks
        elif volatility < 0.01:  # Low volatility
            volatility_multiplier = 0.8  # Tighter stop loss for stable stocks
        
        # Calculate optimal parameters
        stop_loss = default_stop_loss * volatility_multiplier
        take_profit = default_take_profit * volatility_multiplier
        
        # Ensure minimum values
        stop_loss = max(stop_loss, 0.03)  # Minimum 3% stop loss
        take_profit = max(take_profit, 0.05)  # Minimum 5% take profit
        
        # Ensure take profit is always higher than stop loss
        if take_profit <= stop_loss:
            take_profit = stop_loss * 1.5
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'volatility': volatility
        }
    except Exception as e:
        print(f"Error optimizing risk parameters: {e}")
        return {'stop_loss': default_stop_loss, 'take_profit': default_take_profit}

def generate_recommendation(signals, sentiment_score=0.5):
    """
    Generate investment recommendation based on technical signals and sentiment.
    Always returns either BUY or SELL - no more HOLD recommendations.
    
    :param signals: Dictionary of technical signals
    :param sentiment_score: Sentiment score from news (0-1 scale)
    :return: Recommendation dictionary with action and score
    """
    # Initialize positive and negative signals count
    positive_signals = 0
    negative_signals = 0
    total_signals = 0
    reasoning = []
    
    # Check price relative to moving averages
    if signals.get('price_above_sma20', False):
        positive_signals += 1
        total_signals += 1
        reasoning.append("Price is above 20-day moving average")
    else:
        negative_signals += 1
        total_signals += 1
        reasoning.append("Price is below 20-day moving average")
        
    if signals.get('price_above_sma50', False):
        positive_signals += 1
        total_signals += 1
        reasoning.append("Price is above 50-day moving average")
    else:
        negative_signals += 1
        total_signals += 1
        reasoning.append("Price is below 50-day moving average")
    
    # RSI analysis
    rsi = signals.get('rsi', 50)
    if rsi < 30:
        positive_signals += 1.5  # Oversold condition gets higher weight
        total_signals += 1.5
        reasoning.append(f"RSI indicates oversold condition ({rsi:.1f})")
    elif rsi > 70:
        negative_signals += 1.5  # Overbought condition gets higher weight
        total_signals += 1.5
        reasoning.append(f"RSI indicates overbought condition ({rsi:.1f})")
    else:
        # Neutral RSI
        total_signals += 1
        if rsi > 50:
            positive_signals += 0.5
            reasoning.append(f"RSI shows mild bullish momentum ({rsi:.1f})")
        else:
            negative_signals += 0.5
            reasoning.append(f"RSI shows mild bearish momentum ({rsi:.1f})")
    
    # MACD analysis
    macd = signals.get('macd', 0)
    macd_signal = signals.get('macd_signal', 0)
    if macd > macd_signal:
        positive_signals += 1
        total_signals += 1
        reasoning.append("MACD is above signal line (bullish)")
    else:
        negative_signals += 1
        total_signals += 1
        reasoning.append("MACD is below signal line (bearish)")
    
    # Volume analysis
    volume_ratio = signals.get('volume_ratio', 1.0)
    if volume_ratio > 1.2 and signals.get('overall_trend', 0) > 0:
        positive_signals += 1
        total_signals += 1
        reasoning.append(f"Above average volume supporting uptrend ({volume_ratio:.1f}x)")
    elif volume_ratio > 1.2 and signals.get('overall_trend', 0) < 0:
        negative_signals += 1
        total_signals += 1
        reasoning.append(f"Above average volume supporting downtrend ({volume_ratio:.1f}x)")
    
    # Chart patterns (if available)
    patterns = signals.get('patterns', {})
    bullish_patterns = sum(1 for pattern, exists in patterns.items() 
                         if exists and ('bullish' in pattern or 'hammer' in pattern or 
                                       'morning' in pattern or 'white' in pattern))
    bearish_patterns = sum(1 for pattern, exists in patterns.items() 
                         if exists and ('bearish' in pattern or 'hanging' in pattern or 
                                       'evening' in pattern or 'black' in pattern))
    
    if bullish_patterns > 0:
        positive_signals += bullish_patterns * 1.5  # Higher weight for patterns
        total_signals += bullish_patterns * 1.5
        reasoning.append(f"Detected {bullish_patterns} bullish chart pattern(s)")
    
    if bearish_patterns > 0:
        negative_signals += bearish_patterns * 1.5
        total_signals += bearish_patterns * 1.5
        reasoning.append(f"Detected {bearish_patterns} bearish chart pattern(s)")
    
    # Sentiment analysis (from news)
    if sentiment_score > 0.2:
        positive_signals += 1
        total_signals += 1
        reasoning.append(f"Positive news sentiment ({sentiment_score:.2f})")
    elif sentiment_score < -0.1:
        negative_signals += 1
        total_signals += 1
        reasoning.append(f"Negative news sentiment ({sentiment_score:.2f})")
    
    # Calculate final score on scale of -10 to 10
    if total_signals > 0:
        final_score = ((positive_signals - negative_signals) / total_signals) * 10
    else:
        final_score = 0
    
    # Determine recommendation - only BUY or SELL
    if final_score > 0:
        recommendation = "BUY"
        if final_score > 5:
            recommendation = "STRONG BUY"
    else:
        recommendation = "SELL"
        if final_score < -5:
            recommendation = "STRONG SELL"
    
    # Add patterns to the recommendation object
    return {
        "recommendation": recommendation,
        "score": final_score,
        "reasoning": reasoning,
        "patterns": patterns or {},  # Ensure patterns is included
        "signal_strength": abs(final_score) / 10  # 0-1 scale for signal strength
    }

def safe_get(obj, key, default=None):
    """Safely get a value from an object, whether it's a dictionary or has attributes."""
    if obj is None:
        return default
    
    if isinstance(obj, dict):
        return obj.get(key, default)
    
    try:
        # For objects with attributes
        return getattr(obj, key, default)
    except:
        return default

@app.template_filter('safe_get')
def safe_get_filter(obj, key, default='N/A'):
    """Safely get a value from an object whether it's a dict or has attributes."""
    if obj is None:
        return default
    
    if isinstance(obj, dict):
        # Try dictionary access
        return obj.get(key, default)
    
    try:
        # Try attribute access
        return getattr(obj, key, default)
    except:
        return default

@app.template_filter('get_nested')
def get_nested_filter(obj, path, default='N/A'):
    """Get a nested value from an object using dotted path notation."""
    if obj is None:
        return default
        
    parts = path.split('.')
    current = obj
    
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
            
    return current

@app.template_filter('default_dict')
def default_dict_filter(dictionary, key, default_value='N/A'):
    """Get a dictionary value with a default if the key doesn't exist"""
    if dictionary is None or not isinstance(dictionary, dict):
        return default_value
    return dictionary.get(key, default_value)

@app.context_processor
def utility_processor():
    def is_price_above_ma(result, ma_key='sma_20'):
        """Safely check if price is above a moving average"""
        if not result or not isinstance(result, dict):
            return False
            
        price = result.get('price_on_date', None)
        indicators = result.get('technical_indicators', {})
        ma_value = indicators.get(ma_key, None)
        
        if price is None or ma_value is None:
            return False
            
        try:
            return float(price) > float(ma_value)
        except (ValueError, TypeError):
            return False
    
    return dict(is_price_above_ma=is_price_above_ma)

def calculate_chart_patterns(df):
    """
    Calculate chart patterns for technical analysis.
    
    :param df: DataFrame with OHLC data
    :return: DataFrame with added pattern columns
    """
    # Make a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Calculate patterns and add as columns with the prefix 'pattern_'
    try:
        df_copy['pattern_bullish_engulfing'] = calculate_bullish_engulfing(df_copy)
        df_copy['pattern_bearish_engulfing'] = calculate_bearish_engulfing(df_copy)
        df_copy['pattern_doji'] = calculate_doji(df_copy)
        df_copy['pattern_hammer'] = calculate_hammer(df_copy)
        df_copy['pattern_hanging_man'] = calculate_hanging_man(df_copy)
        df_copy['pattern_morning_star'] = calculate_morning_star(df_copy)
        df_copy['pattern_evening_star'] = calculate_evening_star(df_copy)
        df_copy['pattern_three_white_soldiers'] = calculate_three_white_soldiers(df_copy)
        df_copy['pattern_three_black_crows'] = calculate_three_black_crows(df_copy)
        df_copy['pattern_piercing_line'] = calculate_piercing_line(df_copy)
        df_copy['pattern_dark_cloud_cover'] = calculate_dark_cloud_cover(df_copy)
    except Exception as e:
        print(f"Error calculating patterns: {e}")
    
    return df_copy

def init_db():
    """Initialize the database with required tables"""
    conn = get_db_connection()
    
    # Add the entry_date column to portfolio table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS portfolio (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        shares REAL NOT NULL,
        avg_price REAL NOT NULL,
        entry_date TEXT,
        last_updated TEXT
    )
    ''')
    
    # Add missed opportunities table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS missed_opportunities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_id INTEGER NOT NULL,
        max_potential_price REAL,
        max_potential_profit REAL,
        max_potential_profit_percent REAL,
        missed_opportunity REAL,
        missed_opportunity_percent REAL,
        max_price_date TEXT,
        FOREIGN KEY (trade_id) REFERENCES trades (id)
    )
    ''')
    
    # Other existing tables...
    conn.commit()
    conn.close()

@app.route('/missed-opportunities')
def missed_opportunities_debug():
    """Debug view to check missed opportunities data"""
    conn = get_db_connection()
    
    try:
        # Check if table exists
        table_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='missed_opportunities'"
        ).fetchone() is not None
        
        if not table_exists:
            return "<h1>Missed Opportunities Table Not Found</h1><p>The table has not been created yet.</p>"
        
        # Get missed opportunities with trade data
        data = conn.execute('''
            SELECT 
                t.id as trade_id, 
                t.symbol,
                t.date as sell_date,
                t.price as sell_price,
                t.shares,
                t.profit_loss,
                t.profit_loss_percent,
                mo.max_potential_price,
                mo.missed_opportunity_percent,
                mo.max_price_date
            FROM trades t
            LEFT JOIN missed_opportunities mo ON t.id = mo.trade_id
            WHERE t.action = 'SELL'
            ORDER BY t.date DESC
        ''').fetchall()
        
        # Generate simple HTML report
        html = "<h1>Missed Opportunities Debug Report</h1>"
        
        if not data:
            html += "<p>No SELL transactions found. Execute a sell transaction to see missed opportunities.</p>"
        else:
            html += "<table border='1'><tr><th>Trade ID</th><th>Symbol</th><th>Sell Date</th><th>Sell Price</th>"
            html += "<th>Shares</th><th>Actual P/L %</th><th>Potential Max Price</th>"
            html += "<th>Missed Opportunity %</th><th>Best Exit Date</th></tr>"
            
            for row in data:
                missed_pct = row['missed_opportunity_percent'] if row['missed_opportunity_percent'] is not None else 'N/A'
                max_price = row['max_potential_price'] if row['max_potential_price'] is not None else 'N/A'
                max_date = row['max_price_date'] if row['max_price_date'] is not None else 'N/A'
                
                html += f"<tr><td>{row['trade_id']}</td><td>{row['symbol']}</td><td>{row['sell_date']}</td>"
                html += f"<td>${row['sell_price']:.2f}</td><td>{row['shares']}</td>"
                html += f"<td>{row['profit_loss_percent']:.2f}%</td><td>${max_price if max_price != 'N/A' else 'N/A'}</td>"
                html += f"<td>{missed_pct if missed_pct != 'N/A' else 'N/A'}</td><td>{max_date}</td></tr>"
            
            html += "</table>"
        
        return html
    except Exception as e:
        return f"<h1>Error</h1><p>{str(e)}</p>"
    finally:
        conn.close()

# Add this context processor to ensure all templates have access to market data
@app.context_processor
def inject_market_data():
    """Add market data to all templates"""
    return {
        'market_condition': analyze_market_condition(),
        'today_date': datetime.now().strftime('%Y-%m-%d')
    }

@app.route('/debug/sentiment-market')
def debug_sentiment_market():
    """Debug endpoint to check sentiment and market condition functionality"""
    try:
        # Test sentiment analysis
        symbol = request.args.get('symbol', 'AAPL')
        news_articles = fetch_news_from_alpha_vantage(symbol)
        sentiment_score = analyze_sentiment(news_articles)
        
        # Test market condition
        market_condition = analyze_market_condition()
        
        # Generate HTML report
        html = "<h1>Sentiment and Market Condition Debug</h1>"
        html += f"<h2>Sentiment Analysis for {symbol}</h2>"
        html += f"<p>Sentiment Score: {sentiment_score}</p>"
        
        if news_articles:
            html += "<h3>Recent News Articles:</h3>"
            html += "<ul>"
            for article in news_articles[:3]:  # Show first 3 articles
                html += f"<li>{article.get('title', 'No title')} - Sentiment: {article.get('sentiment_score', 'N/A')}</li>"
            html += "</ul>"
        else:
            html += "<p>No news articles found</p>"
            
        html += "<h2>Market Condition Analysis</h2>"
        html += f"<p>Current Market Condition: {market_condition}</p>"
        
        return html
    except Exception as e:
        return f"<h1>Error</h1><p>{str(e)}</p><pre>{traceback.format_exc()}</pre>"

if __name__ == '__main__':
    init_db()  # Initialize database
    app.run(debug=True)
