import sqlite3
from datetime import datetime, timedelta

def get_db_connection():
    """Create a connection to the SQLite database"""
    conn = sqlite3.connect('trading.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with required tables"""
    conn = get_db_connection()
    
    # Create trades table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        date TEXT NOT NULL,
        price REAL NOT NULL,
        shares REAL NOT NULL,
        action TEXT NOT NULL,
        investment REAL NOT NULL,
        stop_loss REAL,
        take_profit REAL,
        profit_loss REAL,
        notes TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create portfolio table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS portfolio (
        symbol TEXT PRIMARY KEY,
        shares REAL NOT NULL,
        avg_price REAL NOT NULL,
        last_updated TEXT NOT NULL
    )
    ''')
    
    # Create analysis history table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS analysis_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        date TEXT NOT NULL,
        recommendation TEXT NOT NULL,
        price REAL NOT NULL,
        technical_score REAL,
        sentiment_score REAL,
        day3_return REAL,
        day5_return REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Insert sample data if tables are empty
    if conn.execute('SELECT COUNT(*) FROM portfolio').fetchone()[0] == 0:
        # Sample portfolio data
        sample_portfolio = [
            ('AAPL', 10, 175.25, datetime.now().strftime('%Y-%m-%d')),
            ('MSFT', 5, 330.75, datetime.now().strftime('%Y-%m-%d')),
            ('AMZN', 8, 140.50, datetime.now().strftime('%Y-%m-%d')),
            ('NVDA', 15, 220.30, datetime.now().strftime('%Y-%m-%d'))
        ]
        conn.executemany('INSERT INTO portfolio VALUES (?, ?, ?, ?)', sample_portfolio)
        
        # Sample trades with some profit/loss
        sample_trades = [
            ('AAPL', (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'), 168.50, 10, 'BUY', 1685.00, 0.05, 0.08, 0, None),
            ('MSFT', (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d'), 320.25, 5, 'BUY', 1601.25, 0.05, 0.08, 0, None),
            ('AMZN', (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'), 135.75, 8, 'BUY', 1086.00, 0.05, 0.08, 0, None),
            ('NVDA', (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'), 210.40, 15, 'BUY', 3156.00, 0.05, 0.08, 0, None),
            ('GOOGL', (datetime.now() - timedelta(days=25)).strftime('%Y-%m-%d'), 128.30, 12, 'BUY', 1539.60, 0.05, 0.08, 0, None),
            ('GOOGL', (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d'), 135.80, 12, 'SELL', 1629.60, 0.0, 0.0, 90.00, 'Profit taking')
        ]
        conn.executemany('''
        INSERT INTO trades (symbol, date, price, shares, action, investment, stop_loss, take_profit, profit_loss, notes) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', sample_trades)
    
    conn.commit()
    conn.close() 