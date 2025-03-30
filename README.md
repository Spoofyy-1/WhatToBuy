# Finance Chat - Stock Analysis & Trading Tool

## Overview
Finance Chat is a comprehensive stock analysis and trading tool that helps investors make informed decisions by combining technical analysis, sentiment analysis, and backtesting capabilities. The application provides personalized investment recommendations, portfolio management features, and detailed market insights.

## Features

### Stock Analysis
- **Technical Analysis**: Calculate and visualize key technical indicators including RSI, MACD, Moving Averages, and support/resistance levels
- **Sentiment Analysis**: Gather and analyze news sentiment from Alpha Vantage API to gauge market perception
- **Chart Patterns**: Detect common chart patterns like bullish/bearish engulfing, doji, hammer, etc.
- **Backtesting**: Test trading strategies with 3-day and 5-day return projections

### Portfolio Management
- **Position Tracking**: Monitor your current stock positions and overall portfolio performance
- **Trade Execution**: Buy and sell stocks with customizable stop-loss and take-profit levels
- **Trade History**: Review past trades with profit/loss calculations
- **Portfolio Allocation**: Visualize how your investments are distributed

### Market Insights
- **Market Condition Analysis**: Assess whether the market is bullish, bearish, or volatile
- **Win Rate Calculation**: Track the success rate of your trading decisions
- **Entry Quality Assessment**: Determine if current price levels represent good entry points

## Setup Instructions

### Prerequisites
- Python 3.7+
- Flask
- SQLite
- Alpha Vantage API key (free tier available)

### Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/finance-chat.git
cd finance-chat
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Set up the Alpha Vantage API key:
   - Sign up for a free API key at [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
   - Update the `ALPHA_VANTAGE_API_KEY` in `data_fetch.py`

4. Initialize the database:
```
python
>>> from database import init_db
>>> init_db()
>>> exit()
```

5. Run the application:
```
python app.py
```

6. Access the web interface at http://127.0.0.1:5000

## Usage Guide

### Analyzing Stocks
1. Navigate to the "Analyze" tab
2. Enter stock symbols separated by spaces (e.g., "AAPL MSFT GOOG")
3. Adjust investment amount, analysis date, stop-loss, and take-profit parameters
4. Click "Analyze" to generate recommendations

### Managing Portfolio
1. Use the "Portfolio" tab to view your current positions
2. Track performance metrics for each position
3. Execute trades directly from the interface
4. Review trade history and performance statistics

### Reading Charts
1. Technical indicators are color-coded for easy interpretation
2. Green/red highlighting indicates positive/negative signals
3. Candlestick charts show price movement patterns
4. Moving averages help identify trends and potential reversals

## API Endpoints

The application provides several API endpoints for developers:

- `/api/stock-data/<symbol>` - Get historical price data for a stock
- `/api/technical/<symbol>` - Get technical indicators for a stock
- `/execute-trade` - Execute a buy/sell trade (POST)
- `/debug/sentiment-market` - View sentiment analysis data

## Troubleshooting

- **API Rate Limits**: Alpha Vantage has rate limits (5 API requests per minute, 500 per day on free tier)
- **Data Availability**: Some stocks may have limited data available
- **Cache Clearing**: Use the "Clear Cache" option if you encounter stale data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Alpha Vantage](https://www.alphavantage.co/) for financial data
- [Chart.js](https://www.chartjs.org/) for data visualization
- [Bootstrap](https://getbootstrap.com/) for UI components 