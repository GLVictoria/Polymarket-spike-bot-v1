
import sqlite3
import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

# Configure logger
logger = logging.getLogger('polymarket_bot')

DB_FILE = "trades.db"

@dataclass
class TradeRecord:
    id: str
    asset_id: str
    entry_price: float
    entry_time: float
    shares: float
    side: str  # 'BUY' or 'SELL'
    status: str  # 'OPEN', 'CLOSED'
    exit_price: Optional[float] = None
    exit_time: Optional[float] = None
    pnl: Optional[float] = None
    reason: Optional[str] = None
    meta: Optional[Dict] = None

class Database:
    def __init__(self, db_file: str = DB_FILE):
        self.db_file = db_file
        self.conn = None
        self.init_db()

    def get_connection(self):
        """Create a database connection if not exists"""
        if self.conn is None:
            try:
                self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
                self.conn.row_factory = sqlite3.Row
            except sqlite3.Error as e:
                logger.error(f"Error connecting to database: {e}")
                return None
        return self.conn

    def init_db(self):
        """Initialize database tables"""
        conn = self.get_connection()
        if not conn:
            return

        try:
            cursor = conn.cursor()
            
            # Create trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    asset_id TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_time REAL NOT NULL,
                    shares REAL NOT NULL,
                    side TEXT NOT NULL,
                    status TEXT NOT NULL,
                    exit_price REAL,
                    exit_time REAL,
                    pnl REAL,
                    reason TEXT,
                    meta TEXT
                )
            ''')
            
            # Create daily_stats table for circuit breaker
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date TEXT PRIMARY KEY,
                    realized_pnl REAL DEFAULT 0.0,
                    trade_count INTEGER DEFAULT 0,
                    last_updated REAL
                )
            ''')
            
            conn.commit()
            logger.info("Database initialized successfully")
        except sqlite3.Error as e:
            logger.error(f"Error initializing database: {e}")

    def add_trade(self, trade: TradeRecord) -> bool:
        """Add a new trade to the database"""
        conn = self.get_connection()
        if not conn:
            return False

        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trades (
                    id, asset_id, entry_price, entry_time, shares, side, status, reason, meta
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.id, trade.asset_id, trade.entry_price, trade.entry_time, 
                trade.shares, trade.side, trade.status, trade.reason, 
                json.dumps(trade.meta) if trade.meta else None
            ))
            conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error adding trade {trade.id}: {e}")
            return False

    def update_trade_exit(self, trade_id: str, exit_price: float, exit_time: float, pnl: float, reason: str = None) -> bool:
        """Update a trade with exit details"""
        conn = self.get_connection()
        if not conn:
            return False

        try:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE trades 
                SET status = 'CLOSED', exit_price = ?, exit_time = ?, pnl = ?, reason = ?
                WHERE id = ?
            ''', (exit_price, exit_time, pnl, reason, trade_id))
            conn.commit()
            
            # Also update daily stats
            self.update_daily_pnl(pnl)
            
            return True
        except sqlite3.Error as e:
            logger.error(f"Error updating trade {trade_id}: {e}")
            return False

    def get_active_trades(self) -> List[TradeRecord]:
        """Get all open trades"""
        conn = self.get_connection()
        if not conn:
            return []

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE status = 'OPEN'")
            rows = cursor.fetchall()
            
            trades = []
            for row in rows:
                meta = json.loads(row['meta']) if row['meta'] else None
                trades.append(TradeRecord(
                    id=row['id'],
                    asset_id=row['asset_id'],
                    entry_price=row['entry_price'],
                    entry_time=row['entry_time'],
                    shares=row['shares'],
                    side=row['side'],
                    status=row['status'],
                    reason=row['reason'],
                    meta=meta
                ))
            return trades
        except sqlite3.Error as e:
            logger.error(f"Error fetching active trades: {e}")
            return []

    def update_daily_pnl(self, pnl_change: float):
        """Update daily PnL statistics"""
        conn = self.get_connection()
        if not conn:
            return

        today = time.strftime('%Y-%m-%d', time.gmtime())
        try:
            cursor = conn.cursor()
            
            # Upsert logic (Insert or Update)
            cursor.execute('''
                INSERT INTO daily_stats (date, realized_pnl, trade_count, last_updated)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(date) DO UPDATE SET
                    realized_pnl = realized_pnl + ?,
                    trade_count = trade_count + 1,
                    last_updated = ?
            ''', (today, pnl_change, time.time(), pnl_change, time.time()))
            
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error updating daily PnL: {e}")

    def get_daily_stats(self) -> Dict[str, Any]:
        """Get stats for today (UTC)"""
        conn = self.get_connection()
        if not conn:
            return {'realized_pnl': 0.0, 'trade_count': 0}

        today = time.strftime('%Y-%m-%d', time.gmtime())
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM daily_stats WHERE date = ?", (today,))
            row = cursor.fetchone()
            
            if row:
                return {
                    'realized_pnl': row['realized_pnl'],
                    'trade_count': row['trade_count'],
                    'last_updated': row['last_updated']
                }
            return {'realized_pnl': 0.0, 'trade_count': 0}
        except sqlite3.Error as e:
            logger.error(f"Error getting daily stats: {e}")
            return {'realized_pnl': 0.0, 'trade_count': 0}

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
