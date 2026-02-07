import json
import uuid
import os
import time
import requests
import threading
import logging
import logging.handlers
import colorlog
from halo import Halo
from datetime import datetime, timedelta
from dotenv import load_dotenv
from web3 import Web3
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import MarketOrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque, defaultdict
from threading import Lock, Event
from concurrent.futures import ThreadPoolExecutor
import signal
import sys
from dataclasses import dataclass
from enum import Enum
import functools
from queue import Queue

# Dashboard imports
from dashboard import start_dashboard, get_dashboard_logger, add_trade_to_history
from database import Database, TradeRecord
from notifications import notification_manager

# Custom Exceptions
class BotError(Exception):
    pass

class ConfigurationError(BotError):
    pass

class NetworkError(BotError):
    pass

class TradingError(BotError):
    pass

class ValidationError(BotError):
    pass

# Custom Types
@dataclass
class TradeInfo:
    entry_price: float
    entry_time: float
    amount: float
    bot_triggered: bool
    trade_id: Optional[str] = None

@dataclass
class PositionInfo:
    eventslug: str
    outcome: str
    asset: str
    avg_price: float
    shares: float
    current_price: float
    initial_value: float
    current_value: float
    pnl: float
    percent_pnl: float
    realized_pnl: float

class TradeType(Enum):
    BUY = "buy"
    SELL = "sell"

# Constants
MAX_RETRIES = 3                 # Number of retries for API calls
BASE_DELAY = 1                  # Base delay for retries
MAX_ERRORS = 5                  # Maximum number of errors before shutting down
API_TIMEOUT = 10                # Timeout for API requests
REFRESH_INTERVAL = 3600         # Refresh interval for API credentials
# COOLDOWN_PERIOD is now loaded from environment variables only (line 142)
THREAD_POOL_SIZE = 3            # Number of threads in the thread pool 
MAX_QUEUE_SIZE = 1000           # Maximum number of items in the queue
THREAD_CHECK_INTERVAL = 5       # Interval for checking thread status
THREAD_RESTART_DELAY = 2        # Delay before restarting a thread
MAX_SPREAD = 0.10               # Maximum allowed spread (10%)

# Load and validate environment variables
load_dotenv(".env")

# Configuration validation
def validate_config() -> None:
    required_vars = {
        "trade_unit": float,
        "slippage_tolerance": float,
        "pct_profit": float,
        "pct_loss": float,
        "cash_profit": float,
        "cash_loss": float,
        "spike_threshold": float,
        "sold_position_time": float,
        "YOUR_PROXY_WALLET": str,
        "BOT_TRADER_ADDRESS": str,
        "USDC_CONTRACT_ADDRESS": str,
        "POLYMARKET_SETTLEMENT_CONTRACT": str,
        "PK": str,
        "holding_time_limit": float,
        "max_concurrent_trades": int,
        "min_liquidity_requirement": float
    }
    
    missing = []
    invalid = []
    
    for var, var_type in required_vars.items():
        value = os.getenv(var)
        if not value:
            missing.append(var)
            continue
        try:
            if var_type == float:
                float(value)
            elif var_type == str:
                str(value)
        except ValueError:
            invalid.append(var)
    
    if missing or invalid:
        error_msg = []
        if missing:
            error_msg.append(f"Missing variables: {', '.join(missing)}")
        if invalid:
            error_msg.append(f"Invalid values for: {', '.join(invalid)}")
        return False, " | ".join(error_msg)
    
    return True, None

# Global configuration - check if valid, but don't crash
CONFIG_VALID, CONFIG_ERROR = validate_config()

# Helper to get config with defaults
def get_config(name: str, default, type_fn=float):
    """Get config value with fallback to default"""
    try:
        value = os.getenv(name)
        if value:
            return type_fn(value)
    except (ValueError, TypeError):
        pass
    return default

# Trading parameters with defaults
TRADE_UNIT = get_config("trade_unit", 3.0)
SLIPPAGE_TOLERANCE = get_config("slippage_tolerance", 0.06)
PCT_PROFIT = get_config("pct_profit", 0.03)
PCT_LOSS = get_config("pct_loss", -0.025)
CASH_PROFIT = get_config("cash_profit", 3.0)
CASH_LOSS = get_config("cash_loss", -3.0)
SPIKE_THRESHOLD = get_config("spike_threshold", 0.01)
SOLD_POSITION_TIME = get_config("sold_position_time", 1800.0)
HOLDING_TIME_LIMIT = get_config("holding_time_limit", 60.0)
PRICE_HISTORY_SIZE = get_config("price_history_size", 120, int)
COOLDOWN_PERIOD = get_config("cooldown_period", 120, int)
KEEP_MIN_SHARES = get_config("keep_min_shares", 1, int)
MAX_CONCURRENT_TRADES = get_config("max_concurrent_trades", 3, int)
MIN_LIQUIDITY_REQUIREMENT = get_config("min_liquidity_requirement", 10.0)

# Web3 and API setup
WEB3_PROVIDER = "https://polygon-rpc.com"

# Safe wallet address loading
def get_checksum_address(env_var: str):
    """Safely get checksum address, return None if not set"""
    try:
        value = os.getenv(env_var)
        if value:
            return Web3.to_checksum_address(value)
    except Exception:
        pass
    return None

YOUR_PROXY_WALLET = get_checksum_address("YOUR_PROXY_WALLET")
BOT_TRADER_ADDRESS = get_checksum_address("BOT_TRADER_ADDRESS")
USDC_CONTRACT_ADDRESS = os.getenv("USDC_CONTRACT_ADDRESS", "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
POLYMARKET_SETTLEMENT_CONTRACT = os.getenv("POLYMARKET_SETTLEMENT_CONTRACT", "0x56C79347e95530c01A2FC76E732f9566dA16E113")
PRIVATE_KEY = os.getenv("PK")

web3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER))
# Setup logging
def setup_logging() -> logging.Logger:
    """Setup enhanced logging configuration with both file and console handlers"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create a logger
    logger = logging.getLogger('polymarket_bot')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(threadName)-12s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s | %(levelname)-8s | %(threadName)-12s | %(name)s | %(message)s%(reset)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white'
        }
    )
    
    # File handler - Rotating file handler with size limit
    file_handler = logging.handlers.RotatingFileHandler(
        'logs/polymarket_bot.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Add dashboard handler for streaming logs to web UI
    try:
        dashboard_handler = get_dashboard_logger()
        logger.addHandler(dashboard_handler)
    except Exception:
        pass  # Dashboard not available yet
    
    return logger

# Initialize logger
logger = setup_logging()

# Add logging decorator for function entry/exit
def log_function_call(logger: logging.Logger):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.debug(f"Entering {func_name}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting {func_name} successfully")
                return result
            except Exception as e:
                logger.error(f"Error in {func_name}: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator

# Add logging context manager
class LoggingContext:
    def __init__(self, logger, level=None, handler=None, close=True):
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()

# Add threading event for price updates
price_update_event = threading.Event()

class ThreadSafeState:
    def __init__(self, max_price_history_size: int = PRICE_HISTORY_SIZE, keep_min_shares: int = KEEP_MIN_SHARES):
        self._price_history_lock = Lock()
        self._active_trades_lock = Lock()
        self._positions_lock = Lock()
        self._asset_pairs_lock = Lock()
        self._recent_trades_lock = Lock()
        self._last_trade_closed_at_lock = Lock()
        self._initialized_assets_lock = Lock()
        self._last_spike_asset_lock = Lock()
        self._last_spike_price_lock = Lock()
        self._counter_lock = Lock()
        self._shutdown_event = Event()
        self._cleanup_complete = Event()
        self._circuit_breaker_lock = Lock()
        self._paused_lock = Lock()
        self._paused = False
        
        self._max_price_history_size = max_price_history_size
        self._keep_min_shares = keep_min_shares
        
        self._price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_price_history_size))
        self._active_trades: Dict[str, TradeInfo] = {}
        self._positions: Dict[str, List[PositionInfo]] = {}
        self._asset_pairs: Dict[str, str] = {}
        self._recent_trades: Dict[str, Dict[str, Optional[float]]] = {}
        self._last_trade_closed_at: float = 0
        self._initialized_assets = set()
        self._last_spike_asset: Optional[str] = None
        self._last_spike_price: Optional[float] = None
        
        self._daily_pnl = 0.0
        self._trading_enabled = True
        self._max_daily_loss = CASH_LOSS * 3  # Example multiplier, configurable later
        
        self.db = Database()
        self._load_active_trades_from_db()
        self._load_daily_stats_from_db()
        
        self._counter = 0

    def _load_daily_stats_from_db(self):
        """Load daily stats for circuit breaker"""
        try:
            stats = self.db.get_daily_stats()
            with self._circuit_breaker_lock:
                self._daily_pnl = stats.get('realized_pnl', 0.0)
                if self._daily_pnl <= self._max_daily_loss:
                    self._trading_enabled = False
                    logger.warning(f"🛑 Circuit Breaker Active on Startup! Daily PnL: ${self._daily_pnl:.2f}")
            logger.info(f"💰 Daily PnL Loaded: ${self._daily_pnl:.2f}")
        except Exception as e:
            logger.error(f"❌ Failed to load daily stats: {e}")

    def update_pnl(self, amount: float):
        """Update daily PnL and check circuit breaker"""
        with self._circuit_breaker_lock:
            self._daily_pnl += amount
            if self._daily_pnl <= self._max_daily_loss:
                if self._trading_enabled:
                    self._trading_enabled = False
                    msg = f"🛑 Circuit Breaker Triggered!\nDaily PnL: ${self._daily_pnl:.2f}"
                    logger.warning(msg)
                    notification_manager.send_error(msg)
    
    def is_trading_enabled(self) -> bool:
        with self._circuit_breaker_lock:
            return self._trading_enabled

    def _load_active_trades_from_db(self):
        """Load active trades from database on startup"""
        try:
            active_trades = self.db.get_active_trades()
            with self._active_trades_lock:
                for trade in active_trades:
                    # Convert TradeRecord to TradeInfo
                    self._active_trades[trade.asset_id] = TradeInfo(
                        entry_price=trade.entry_price,
                        entry_time=trade.entry_time,
                        amount=trade.shares,
                        bot_triggered=True,  # Assume loaded trades are bot-triggered
                        trade_id=trade.id
                    )
            if active_trades:
                logger.info(f"🔄 Loaded {len(active_trades)} active trades from database")
        except Exception as e:
            logger.error(f"❌ Failed to load active trades from DB: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self) -> None:
        if not self._cleanup_complete.is_set():
            self.shutdown()
            with self._price_history_lock:
                self._price_history.clear()
            with self._active_trades_lock:
                self._active_trades.clear()
            with self._positions_lock:
                self._positions.clear()
            with self._asset_pairs_lock:
                self._asset_pairs.clear()
            with self._recent_trades_lock:
                self._recent_trades.clear()
            self._cleanup_complete.set()

    def increment_counter(self) -> int:
        with self._counter_lock:
            self._counter += 1
            return self._counter

    def reset_counter(self) -> None:
        with self._counter_lock:
            self._counter = 0

    def get_counter(self) -> int:
        with self._counter_lock:
            return self._counter

    def shutdown(self) -> None:
        self._shutdown_event.set()

    def is_shutdown(self) -> bool:
        return self._shutdown_event.is_set()

    def wait_for_cleanup(self, timeout: Optional[float] = None) -> bool:
        return self._cleanup_complete.wait(timeout)

    def get_price_history(self, asset_id: str) -> deque:
        with self._price_history_lock:
            return self._price_history.get(asset_id, deque())

    def add_price(self, asset_id: str, timestamp: float, price: float, eventslug: str, outcome: str) -> None:
        with self._price_history_lock:
            if not isinstance(asset_id, str):
                raise ValidationError(f"Invalid asset_id type: {type(asset_id)}")
            if asset_id not in self._price_history:
                self._price_history[asset_id] = deque(maxlen=self._max_price_history_size)
            self._price_history[asset_id].append((timestamp, price, eventslug, outcome))

    def get_active_trades(self) -> Dict[str, TradeInfo]:
        with self._active_trades_lock:
            return dict(self._active_trades)

    def add_active_trade(self, asset_id: str, trade_info: TradeInfo) -> None:
        with self._active_trades_lock:
            self._active_trades[asset_id] = trade_info

    def remove_active_trade(self, asset_id: str) -> None:
        with self._active_trades_lock:
            self._active_trades.pop(asset_id, None)

    def get_positions(self) -> Dict[str, List[PositionInfo]]:
        with self._positions_lock:
            return dict(self._positions)

    def update_positions(self, new_positions: Dict[str, List[PositionInfo]]) -> None:
        """Update positions with proper validation and error handling"""
        if new_positions is None:
            logger.warning("⚠️ Attempted to update positions with None")
            return
        
        if not isinstance(new_positions, dict):
            logger.error(f"❌ Invalid positions type: {type(new_positions)}")
            return
        
        try:
            with self._positions_lock:
                # Validate each position before updating
                valid_positions = {}
                for event_id, positions in new_positions.items():
                    if not isinstance(positions, list):
                        logger.warning(f"⚠️ Invalid positions list for event {event_id}")
                        continue
                        
                    valid_positions[event_id] = []
                    for pos in positions:
                        if not isinstance(pos, PositionInfo):
                            logger.warning(f"⚠️ Invalid position type for event {event_id}")
                            continue
                            
                        # Validate position data
                        if not pos.asset or not pos.eventslug or not pos.outcome:
                            logger.warning(f"⚠️ Missing required fields in position for event {event_id}")
                            continue
                            
                        if pos.shares < 0 or pos.avg_price < 0 or pos.current_price < 0:
                            logger.warning(f"⚠️ Invalid numeric values in position for event {event_id}")
                            continue
                            
                        valid_positions[event_id].append(pos)
                
                # Only update if we have valid positions
                if valid_positions:
                    self._positions = valid_positions
                    logger.info(f"✅ Updated positions: {len(valid_positions)} events")
                else:
                    logger.warning("⚠️ No valid positions to update")
                
        except Exception as e:
            logger.error(f"❌ Error updating positions: {str(e)}")
            # Keep old positions if update fails
            return

    def get_asset_pair(self, asset_id: str) -> Optional[str]:
        with self._asset_pairs_lock:
            return self._asset_pairs.get(asset_id)

    def add_asset_pair(self, asset1: str, asset2: str) -> None:
        with self._asset_pairs_lock:
            self._asset_pairs[asset1] = asset2
            self._asset_pairs[asset2] = asset1
            self._initialized_assets.add(asset1)
            self._initialized_assets.add(asset2)

    def is_initialized(self) -> bool:
        with self._initialized_assets_lock:
            return len(self._initialized_assets) > 0

    def update_recent_trade(self, asset_id: str, trade_type: TradeType) -> None:
        with self._recent_trades_lock:
            if asset_id not in self._recent_trades:
                self._recent_trades[asset_id] = {"buy": None, "sell": None}
            self._recent_trades[asset_id][trade_type.value] = time.time()

    def get_last_trade_time(self) -> float:
        with self._last_trade_closed_at_lock:
            return self._last_trade_closed_at

    def set_last_trade_time(self, timestamp: float) -> None:
        with self._last_trade_closed_at_lock:
            self._last_trade_closed_at = timestamp

    def get_last_spike_info(self) -> Tuple[Optional[str], Optional[float]]:
        with self._last_spike_asset_lock, self._last_spike_price_lock:
            return self._last_spike_asset, self._last_spike_price

    def set_last_spike_info(self, asset: str, price: float) -> None:
        with self._last_spike_asset_lock, self._last_spike_price_lock:
            self._last_spike_asset = asset
            self._last_spike_price = price

    # def update_daily_pnl(self, pnl: float) -> None:
    #     with self._circuit_breaker_lock:
    #         self._daily_pnl += pnl

    def toggle_pause(self) -> bool:
        """Toggle pause state. Returns new state."""
        with self._paused_lock:
            self._paused = not self._paused
            return self._paused

    def is_paused(self) -> bool:
        """Check if bot is paused"""
        with self._paused_lock:
            return self._paused
    #         logger.info(f"📊 Daily PnL updated: ${self._daily_pnl:.2f}")
            
    #         # Check circuit breaker conditions
    #         if self._daily_pnl < self._max_daily_loss:
    #             logger.error(f"🔴 Circuit breaker triggered: Daily loss limit reached (${self._daily_pnl:.2f})")
    #             self._trading_enabled = False
    #         elif self._daily_pnl < self._max_drawdown:
    #             logger.error(f"🔴 Circuit breaker triggered: Maximum drawdown reached (${self._daily_pnl:.2f})")
    #             self._trading_enabled = False

    # def is_trading_enabled(self) -> bool:
    #     with self._circuit_breaker_lock:
    #         return self._trading_enabled

    # def reset_daily_pnl(self) -> None:
    #     with self._circuit_breaker_lock:
    #         self._daily_pnl = 0.0
    #         self._trading_enabled = True
    #         logger.info("🔄 Daily PnL reset and trading enabled")

# Initialize ClobClient with retry mechanism
def initialize_clob_client(max_retries: int = 3) -> ClobClient:
    global PRIVATE_KEY, YOUR_PROXY_WALLET
    
    # Ensure we have the latest keys
    if not PRIVATE_KEY or not YOUR_PROXY_WALLET:
        PRIVATE_KEY = os.getenv("PK")
        val = os.getenv("YOUR_PROXY_WALLET")
        if val:
            YOUR_PROXY_WALLET = Web3.to_checksum_address(val)

    for attempt in range(max_retries):
        try:
            logger.info(f"🔑 Initializing CLOB client with proxy: {YOUR_PROXY_WALLET}")
            c = ClobClient(
                host="https://clob.polymarket.com",
                key=PRIVATE_KEY,
                chain_id=137,
                signature_type=2,
                funder=YOUR_PROXY_WALLET
            )
            api_creds = c.create_or_derive_api_creds()
            c.set_api_creds(api_creds)
            return c
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"❌ Failed to initialize CLOB client: {e}")
                raise
            logger.warning(f"Failed to initialize ClobClient (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError("Failed to initialize ClobClient after maximum retries")

client = None

# API functions with retry mechanism
def fetch_positions_with_retry(max_retries: int = MAX_RETRIES) -> Dict[str, List[PositionInfo]]:
    for attempt in range(max_retries):
        try:
            url = f"https://data-api.polymarket.com/positions?user={YOUR_PROXY_WALLET}"
            logger.info(f"🔄 Fetching positions from {url} (attempt {attempt + 1}/{max_retries})")
            
            response = requests.get(url, timeout=API_TIMEOUT)
            logger.info(f"📡 API Response Status: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"❌ API Error: {response.status_code} - {response.text}")
                raise NetworkError(f"API returned status code {response.status_code}")
            
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, list):
                logger.error(f"❌ Invalid response format: {type(data)}")
                logger.error(f"Response content: {data}")
                raise ValidationError(f"Invalid response format from API: {type(data)}")
            
            if not data:
                logger.warning("⚠️ No positions found in API response. Waiting for positions...")
                return {}
                
            positions: Dict[str, List[PositionInfo]] = {}
            for pos in data:
                event_id = pos.get("conditionId") or pos.get("eventId") or pos.get("marketId")
                if not event_id:
                    logger.warning(f"⚠️ Skipping position with no event ID: {pos}")
                    continue
                    
                if event_id not in positions:
                    positions[event_id] = []
                    
                try:
                    position_info = PositionInfo(
                        eventslug=pos.get("eventSlug", ""),
                        outcome=pos.get("outcome", ""),
                        asset=pos.get("asset", ""),
                        avg_price=float(pos.get("avgPrice", 0)),
                        shares=float(pos.get("size", 0)),
                        current_price=float(pos.get("curPrice", 0)),
                        initial_value=float(pos.get("initialValue", 0)),
                        current_value=float(pos.get("currentValue", 0)),
                        pnl=float(pos.get("cashPnl", 0)),
                        percent_pnl=float(pos.get("percentPnl", 0)),
                        realized_pnl=float(pos.get("realizedPnl", 0))
                    )
                    positions[event_id].append(position_info)
                    logger.debug(f"✅ Added position: {position_info}")
                except (ValueError, TypeError) as e:
                    logger.error(f"❌ Error parsing position data: {e}")
                    logger.error(f"Problematic position data: {pos}")
                    continue
            
            logger.info(f"✅ Successfully fetched {len(positions)} positions")
            return positions
            
        except requests.RequestException as e:
            logger.error(f"❌ Network error in fetch_positions (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                raise NetworkError(f"Failed to fetch positions after {max_retries} attempts: {e}")
            time.sleep(2 ** attempt)
        except (ValueError, ValidationError) as e:
            logger.error(f"❌ Validation error in fetch_positions (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                raise ValidationError(f"Invalid data received from API: {e}")
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"❌ Unexpected error in fetch_positions (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                raise NetworkError(f"Failed to fetch positions after {max_retries} attempts: {e}")
            time.sleep(2 ** attempt)
    
    raise NetworkError("Failed to fetch positions after maximum retries")

def ensure_usdc_allowance(required_amount: float) -> bool:
    """Ensure USDC allowance with proper error handling"""
    max_retries = MAX_RETRIES
    base_delay = BASE_DELAY
    
    for attempt in range(max_retries):
        try:
            contract = web3.eth.contract(address=USDC_CONTRACT_ADDRESS, abi=[
                {"constant": True, "inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}],
                 "name": "allowance", "outputs": [{"name": "", "type": "uint256"}],
                 "payable": False, "stateMutability": "view", "type": "function"},
                {"constant": False, "inputs": [{"name": "spender", "type": "address"}, {"name": "value", "type": "uint256"}],
                 "name": "approve", "outputs": [{"name": "", "type": "bool"}],
                 "payable": False, "stateMutability": "nonpayable", "type": "function"}
            ])

            current_allowance = contract.functions.allowance(BOT_TRADER_ADDRESS , POLYMARKET_SETTLEMENT_CONTRACT).call()
            logger.info(f"current_allowance: {current_allowance}")
            required_amount_with_buffer = int(required_amount * 1.1 * 10**6)
            
            if current_allowance >= required_amount_with_buffer:
                return True

            logger.info(f"🔄 Approving USDC allowance... (attempt {attempt + 1}/{max_retries})")
            
            new_allowance = max(current_allowance, required_amount_with_buffer)
            logger.info(f"new_allowance: {new_allowance}")
            txn = contract.functions.approve(POLYMARKET_SETTLEMENT_CONTRACT, new_allowance).build_transaction({
                "from": BOT_TRADER_ADDRESS,
                "gas": 200000,
                "gasPrice": web3.eth.gas_price,
                "nonce": web3.eth.get_transaction_count(BOT_TRADER_ADDRESS),
                "chainId": 137
            })
            
            signed_txn = web3.eth.account.sign_transaction(txn, private_key=PRIVATE_KEY)
            tx_hash = web3.eth.send_raw_transaction(signed_txn.raw_transaction)
            receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status == 1:
                logger.info(f"✅ USDC allowance updated: {tx_hash.hex()}")
                return True
            else:
                raise TradingError(f"USDC allowance update failed: {tx_hash.hex()}")
                
        except Exception as e:
            if attempt == max_retries - 1:
                raise TradingError(f"Failed to update USDC allowance: {e}")
            logger.error(f"⚠️ Error in USDC allowance update (attempt {attempt + 1}): {e}")
            time.sleep(base_delay * (2 ** attempt))
    
    return False

def refresh_api_credentials() -> bool:
    """Refresh API credentials with proper error handling"""
    try:
        api_creds = client.create_or_derive_api_creds()
        client.set_api_creds(api_creds)
        logger.info("✅ API credentials refreshed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to refresh API credentials: {str(e)}")
        return False

def get_min_ask_data(asset: str) -> Optional[Dict[str, Any]]:
    try:
        order = client.get_order_book(asset)
        if order.asks:
            buy_price = client.get_price(asset, "BUY")
            min_ask_price = order.asks[-1].price
            min_ask_size = order.asks[-1].size
            logger.info(f"min_ask_price: {min_ask_price}, min_ask_size: {min_ask_size}")
            return {
                "buy_price": buy_price,
                "min_ask_price": min_ask_price,
                "min_ask_size": min_ask_size
            }
        else:
            logger.error(f"❌ No ask data found for {asset}")
            return None
    except Exception as e:
        logger.error(f"❌ Failed to get ask data for {asset}: {str(e)}")
        return None

def get_max_bid_data(asset: str) -> Optional[Dict[str, Any]]:
    try:
        order = client.get_order_book(asset)
        if order.bids:
            sell_price = client.get_price(asset, "SELL")
            max_bid_price = order.bids[-1].price
            max_bid_size = order.bids[-1].size
            logger.info(f"max_bid_price: {max_bid_price}, max_bid_size: {max_bid_size}")
            return {
                "sell_price": sell_price,
                "max_bid_price": max_bid_price,
                "max_bid_size": max_bid_size
            }
        else:
            logger.error(f"❌ No bid data found for {asset}")
            return None
    except Exception as e:
        logger.error(f"❌ Failed to get bid data for {asset}: {str(e)}")
        return None

def check_usdc_balance(usdc_needed: float) -> bool:
    try:
        usdc_contract = web3.eth.contract(address=USDC_CONTRACT_ADDRESS, abi=[
            {"constant": True, "inputs": [{"name": "account", "type": "address"}],
             "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}],
             "payable": False, "stateMutability": "view", "type": "function"}
        ])
        usdc_balance = usdc_contract.functions.balanceOf(YOUR_PROXY_WALLET).call() / 10**6
        
        logger.info(f"💵 USDC Balance: ${usdc_balance:.2f}, Required: ${usdc_needed:.2f}")
        
        if usdc_balance < usdc_needed:
            logger.warning(f"❌ Insufficient USDC balance. Required: ${usdc_needed:.2f}, Available: ${usdc_balance:.2f}")
            return False
        return True

    except Exception as e:
        logger.error(f"❌ Failed to check USDC balance: {str(e)}")
        return False

@log_function_call(logger)
def place_buy_order(state: ThreadSafeState, asset: str, reason: str) -> bool:
    try:
        # Check circuit breaker
        if not state.is_trading_enabled():
            logger.warning("🔒 Trading disabled due to circuit breaker")
            return False

        # Check maximum concurrent trades
        active_trades = state.get_active_trades()
        logger.info(f"active_trades----------------------------------------------->{active_trades}")
        if len(active_trades) >= MAX_CONCURRENT_TRADES:
            logger.warning(f"🔒 Maximum concurrent trades limit reached ({len(active_trades)}/{MAX_CONCURRENT_TRADES})")
            return False

        # Check USDC balance and calculate position size
        usdc_contract = web3.eth.contract(address=USDC_CONTRACT_ADDRESS, abi=[
            {"constant": True, "inputs": [{"name": "account", "type": "address"}],
             "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}],
             "payable": False, "stateMutability": "view", "type": "function"}
        ])
        usdc_balance = usdc_contract.functions.balanceOf(YOUR_PROXY_WALLET).call() / 10**6
        if not usdc_balance:
            return False
            
        
        max_retries = MAX_RETRIES
        base_delay = BASE_DELAY

        for attempt in range(max_retries):
            try:
                current_price = get_current_price(state, asset)
                if current_price is None:
                    raise TradingError(f"Failed to get current price for {asset}")

                min_ask_data = get_min_ask_data(asset)
                if min_ask_data is None:
                    logger.warning(f"❌ The {asset} is not tradable, Skipping...")
                    return False

                min_ask_price = float(min_ask_data["min_ask_price"])
                min_ask_size = float(min_ask_data["min_ask_size"])
                
                # Check liquidity requirement
                if min_ask_size * min_ask_price < MIN_LIQUIDITY_REQUIREMENT:
                    logger.warning(f"🔒 Insufficient liquidity for {asset}. Required: ${MIN_LIQUIDITY_REQUIREMENT}, Available: ${min_ask_size * min_ask_price:.2f}")
                    return False

                # Calculate slippage as percentage
                slippage_pct = (min_ask_price - current_price) / current_price if current_price > 0 else 0
                if slippage_pct > SLIPPAGE_TOLERANCE:
                    logger.warning(f"🔐 Slippage tolerance exceeded for {asset}. Slippage: {slippage_pct:.2%}, Tolerance: {SLIPPAGE_TOLERANCE:.2%}")
                    return False

                # Check Spread
                try:
                    max_bid_data = get_max_bid_data(asset)
                    if max_bid_data:
                        bid_price = float(max_bid_data["max_bid_price"])
                        if bid_price > 0:
                            spread = (min_ask_price - bid_price) / bid_price
                            if spread > MAX_SPREAD:
                                logger.warning(f"📉 Spread too high for {asset}: {spread:.2%}. Limit: {MAX_SPREAD:.2%}")
                                return False
                except Exception as e:
                    logger.warning(f"⚠️ Failed to check spread for {asset}: {e}")



                # Calculate position size based on account balance
                amount_in_dollars = min(TRADE_UNIT, min_ask_size * min_ask_price)
                
                if not check_usdc_balance(amount_in_dollars):
                    raise TradingError(f"Insufficient USDC balance for {asset}")
                
                if not ensure_usdc_allowance(amount_in_dollars):
                    raise TradingError(f"Failed to ensure USDC allowance for {asset}")

                order_args = MarketOrderArgs(
                    token_id=str(asset),
                    amount=float(amount_in_dollars),
                    side=BUY,
                )
                signed_order = client.create_market_order(order_args)
                response = client.post_order(signed_order, OrderType.FOK)
                if response.get("success"):
                    filled = response.get("data", {}).get("filledAmount", amount_in_dollars)
                    logger.info(f"🛒 [{reason}] Order placed: BUY {filled:.4f} shares of {asset} at ${min_ask_price:.4f}")
                    
                    # Generate trade ID
                    trade_id = str(uuid.uuid4())
                    
                    trade_info = TradeInfo(
                        entry_price=min_ask_price,
                        entry_time=time.time(),
                        amount=amount_in_dollars,
                        bot_triggered=True,
                        trade_id=trade_id
                    )
                    
                    state.update_recent_trade(asset, TradeType.BUY)
                    state.add_active_trade(asset, trade_info)
                    state.set_last_trade_time(time.time())
                    
                    # Save to DB
                    if hasattr(state, 'db'):
                        est_shares = amount_in_dollars / min_ask_price if min_ask_price > 0 else 0
                        record = TradeRecord(
                            id=trade_id,
                            asset_id=asset,
                            entry_price=min_ask_price,
                            entry_time=time.time(),
                            shares=est_shares,
                            side='BUY',
                            status='OPEN',
                            reason=reason,
                            meta={'amount_usdc': amount_in_dollars}
                        )
                        state.db.add_trade(record)
                    
                    # Track trade for dashboard
                    add_trade_to_history("BUY", asset, min_ask_price, amount_in_dollars, reason)
                    notification_manager.send_trade_alert("BUY", asset, min_ask_price, amount_in_dollars, reason)
                    return True
                else:
                    error_msg = response.get("error", "Unknown error")
                    raise TradingError(f"Failed to place BUY order for {asset}: {error_msg}")

            except TradingError as e:
                logger.error(f"❌ Trading error in BUY order for {asset}: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(base_delay * (2 ** attempt))
            except Exception as e:
                logger.error(f"❌ Unexpected error in BUY order for {asset}: {str(e)}")
                if attempt == max_retries - 1:
                    raise TradingError(f"Failed to process BUY order after {max_retries} attempts: {e}")
                time.sleep(base_delay * (2 ** attempt))

        return False
    except Exception as e:
        logger.error(f"❌ Error placing BUY order for {asset}: {str(e)}", exc_info=True)
        raise

def place_sell_order(state: ThreadSafeState, asset: str, reason: str) -> bool:
    try:
        # # Check circuit breaker
        # if not state.is_trading_enabled():
        #     logger.warning("🔒 Trading disabled due to circuit breaker")
        #     return False

        max_retries = MAX_RETRIES
        base_delay = BASE_DELAY

        for attempt in range(max_retries):
            try:
                logger.info(f"🔄 Order attempt {attempt + 1}/{max_retries} for SELL {asset}")
                
                current_price = get_current_price(state,asset)
                if current_price is None:
                    raise TradingError(f"Failed to get current price for {asset}")

                max_bid_data = get_max_bid_data(asset)
                if max_bid_data is None:
                    logger.warning(f"❌ The {asset} is not tradable, Skipping...")
                    return False

                max_bid_price = float(max_bid_data["max_bid_price"])
                max_bid_size = float(max_bid_data["max_bid_size"])

                # Initialize variables with defaults to avoid undefined variable errors
                balance = 0
                avg_price = 0
                sell_amount_in_shares = 0
                position_found = False
                
                positions = state.get_positions()
                for event_id, item in positions.items():
                    for position in item:
                        if position.asset == asset:
                            balance = position.shares
                            avg_price = position.avg_price
                            sell_amount_in_shares = balance - KEEP_MIN_SHARES
                            position_found = True
                            break
                    if position_found:
                        break

                if not position_found:
                    logger.warning(f"🙄 No position found for {asset}, Skipping...")
                    return False

                if sell_amount_in_shares < 1:
                    logger.warning(f"🙄 No shares to sell for {asset} (balance: {balance}, min keep: {KEEP_MIN_SHARES}), Skipping...")
                    return False

                slippage = current_price - max_bid_price
                # Fixed: avg_price > max_bid_price means we're selling below entry = LOSS
                if avg_price > max_bid_price:
                    loss_amount = sell_amount_in_shares * (avg_price - max_bid_price)
                    logger.info(f"balance: {balance}, slippage: {slippage}----You will LOSE ${loss_amount:.2f}")
                else:
                    profit_amount = sell_amount_in_shares * (max_bid_price - avg_price)
                    logger.info(f"balance: {balance}, slippage: {slippage}----You will EARN ${profit_amount:.2f}")

                order_args = MarketOrderArgs(
                    token_id=str(asset),
                    amount=float(sell_amount_in_shares),
                    side=SELL,
                )
                signed_order = client.create_market_order(order_args)
                response = client.post_order(signed_order, OrderType.FOK)
                if response.get("success"):
                    filled = response.get("data", {}).get("filledAmount", sell_amount_in_shares)
                    logger.info(f"🛒 [{reason}] Order placed: SELL {filled:.4f} shares of {asset}")
                    
                    # Close trade in DB
                    if hasattr(state, 'db'):
                        try:
                            # Use active_trades lock via getter/setter? Or just accept slight race condition
                            # Better: get trade info before removing
                            active_trades_snapshot = state.get_active_trades()
                            if asset in active_trades_snapshot:
                                trade_info = active_trades_snapshot[asset]
                                if trade_info.trade_id:
                                    pnl = (max_bid_price - avg_price) * sell_amount_in_shares
                                    state.db.update_trade_exit(
                                        trade_id=trade_info.trade_id,
                                        exit_price=max_bid_price,
                                        exit_time=time.time(),
                                        pnl=pnl,
                                        reason=reason
                                    )
                                    # Update in-memory PnL and check circuit breaker
                                    state.update_pnl(pnl)
                        except Exception as e:
                            logger.error(f"Failed to update DB for SELL: {e}")

                    state.update_recent_trade(asset, TradeType.SELL)
                    state.remove_active_trade(asset)
                    state.set_last_trade_time(time.time())
                    
                    # Track trade for dashboard
                    add_trade_to_history("SELL", asset, max_bid_price, sell_amount_in_shares, reason)
                    value = max_bid_price * sell_amount_in_shares
                    notification_manager.send_trade_alert("SELL", asset, max_bid_price, value, reason)
                    return True
                else:
                    error_msg = response.get("error", "Unknown error")
                    raise TradingError(f"Failed to place SELL order for {asset}: {error_msg}")

            except TradingError as e:
                logger.error(f"❌ Trading error in SELL order for {asset}: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(base_delay * (2 ** attempt))
            except Exception as e:
                logger.error(f"❌ Unexpected error in SELL order for {asset}: {str(e)}")
                if attempt == max_retries - 1:
                    raise TradingError(f"Failed to process SELL order after {max_retries} attempts: {e}")
                time.sleep(base_delay * (2 ** attempt))

        return False
    except Exception as e:
        logger.error(f"❌ Error placing SELL order for {asset}: {str(e)}")
        raise

def is_recently_bought(state: ThreadSafeState, asset_id: str) -> bool:
    with state._recent_trades_lock:
        if asset_id not in state._recent_trades or state._recent_trades[asset_id]["buy"] is None:
            return False
        now = time.time()
        time_since_buy = now - state._recent_trades[asset_id]["buy"]
        return time_since_buy < COOLDOWN_PERIOD

def is_recently_sold(state: ThreadSafeState, asset_id: str) -> bool:
    with state._recent_trades_lock:
        if asset_id not in state._recent_trades or state._recent_trades[asset_id]["sell"] is None:
            return False
        now = time.time()
        time_since_sell = now - state._recent_trades[asset_id]["sell"]
        return time_since_sell < COOLDOWN_PERIOD

def find_position_by_asset(positions: dict, asset_id: str) -> Optional[PositionInfo]:
    for event_positions in positions.values():
        for position in event_positions:
            if position.asset == asset_id:
                return position
    return None

class ThreadManager:
    def __init__(self, state: ThreadSafeState):
        self.state = state
        self.threads = {}
        self.thread_queues = {}
        self.executor = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)
        self.running = True
        
    def start_thread(self, name: str, target: callable) -> None:
        if name in self.threads and self.threads[name].is_alive():
            return
            
        queue = Queue(maxsize=MAX_QUEUE_SIZE)
        self.thread_queues[name] = queue
        
        def thread_wrapper():
            error_count = 0
            consecutive_errors = 0
            while self.running and not self.state.is_shutdown():
                try:
                    target(self.state)
                    error_count = 0  # Reset error count on successful iteration
                    consecutive_errors = 0  # Reset consecutive errors
                    time.sleep(0.1)  # Small sleep to prevent CPU spinning
                except Exception as e:
                    error_count += 1
                    consecutive_errors += 1
                    logger.error(f"❌ Error in {name} thread: {str(e)}")
                    
                    if consecutive_errors >= MAX_ERRORS:
                        logger.error(f"❌ Too many consecutive errors in {name} thread. Restarting...")
                        time.sleep(THREAD_RESTART_DELAY)
                        consecutive_errors = 0  # Reset after restart delay
                    else:
                        time.sleep(1)  # Sleep between retries
        
        thread = threading.Thread(
            target=thread_wrapper,
            daemon=True,
            name=name
        )
        thread.start()
        self.threads[name] = thread
        logger.info(f"✅ Started thread: {name}")
        
    def stop(self) -> None:
        """Stop all threads gracefully"""
        self.running = False
        for thread in self.threads.values():
            if thread.is_alive():
                thread.join(timeout=5)
        self.executor.shutdown(wait=True)

def update_price_history(state: ThreadSafeState) -> None:
    last_log_time = time.time()
    update_count = 0
    initial_update = True
    
    while not state.is_shutdown():
        try:
            logger.info("🔄 Updating price history")
            start_time = time.time()
            
            now = time.time()
            positions = fetch_positions_with_retry()
            
            if not positions:
                time.sleep(5)
                continue
                
            state.update_positions(positions)
            
            price_updated = False
            current_time = time.time()
            price_updates = []
            
            for event_id, assets in positions.items():
                for asset in assets:
                    try:
                        eventslug = asset.eventslug
                        outcome = asset.outcome
                        asset_id = asset.asset
                        price = asset.current_price
                        
                        if not asset_id:
                            continue
                            
                        state.add_price(asset_id, now, price, eventslug, outcome)
                        update_count += 1
                        price_updated = True
                        
                        # Only log significant price changes
                        price_updates.append(f"                                               💸 {outcome} in {eventslug}: ${price:.4f}")
                            
                    except IndexError as e:
                        # Handle deque index out of range error
                        logger.debug(f"⏳ Building price history for {asset_id} - {eventslug}")
                        continue
                    except Exception as e:
                        logger.error(f"❌ Error updating price for asset {asset_id}: {str(e)}")
                        continue
            
            # Log price updates every 5 seconds
            if current_time - last_log_time >= 5:
                logger.info("📊 Price Updates:\n" + "\n".join(price_updates))
                last_log_time = current_time
                    
            if price_updated:
                price_update_event.set()
                if initial_update:
                    initial_update = False
                    logger.info("✅ Initial price data population complete")
            
            # Log summary every 1 minute
            if update_count >= 60:
                logger.info(f"📊 Price Update Summary | Updates: {update_count} | Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                update_count = 0
            
            # Ensure we don't run too fast
            elapsed = time.time() - start_time
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
                
        except Exception as e:
            logger.error(f"❌ Error in price update: {str(e)}")
            time.sleep(1)

def detect_and_trade(state: ThreadSafeState) -> None:
    last_log_time = time.time()
    scan_count = 0
    
    while not state.is_shutdown():
        # Check if paused
        if state.is_paused():
            # Log only once every 30 seconds
            if time.time() - last_log_time >= 30:
                logger.info("⏸️ Bot is PAUSED. Scanning skipped.")
                last_log_time = time.time()
            time.sleep(1)
            continue
        try:
            # Wait for price update with timeout
            if price_update_event.wait(timeout=1.0):
                price_update_event.clear()
                
                # Ensure we have some price history before proceeding
                if not any(state.get_price_history(asset_id) for asset_id in state._price_history.keys()):
                    logger.debug("⏳ Waiting for price history to be populated...")
                    continue
                
                positions_copy = state.get_positions()
                scan_count += 1
                
                # Log scan progress every 5 seconds
                current_time = time.time()
                if current_time - last_log_time >= 5:
                    logger.info(f"🔍 Scanning Markets | Scan #{scan_count} | Active Positions: {len(positions_copy)}")
                    last_log_time = current_time
                
                for asset_id in list(state._price_history.keys()):
                    try:
                        history = state.get_price_history(asset_id)
                        # Need at least 5 price points for spike detection
                        if len(history) < 5:
                            continue

                        # Use recent prices for spike detection (last 5-10 seconds)
                        # Compare current price to price from ~5 updates ago (5 seconds)
                        recent_lookback = min(5, len(history) - 1)
                        old_price = history[-recent_lookback - 1][1]
                        new_price = history[-1][1]
                        
                        # Skip if either price is zero to prevent division by zero
                        if old_price == 0 or new_price == 0:
                            logger.warning(f"⚠️ Skipping asset {asset_id} due to zero price - Old: ${old_price:.4f}, New: ${new_price:.4f}")
                            continue
                            
                        delta = (new_price - old_price) / old_price

                        if abs(delta) > SPIKE_THRESHOLD:
                            # Skip prices outside tradeable range
                            if new_price < 0.20 or new_price > 0.80:
                                logger.debug(f"⚠️ Skipping {asset_id} - price ${new_price:.4f} outside range [0.20, 0.80]")
                                continue

                            opposite = state.get_asset_pair(asset_id)
                            if not opposite:
                                logger.debug(f"⚠️ No opposite pair found for {asset_id}")
                                continue

                            if delta > 0 and not is_recently_bought(state, asset_id):
                                logger.info(f"🟨 Spike Detected | Asset: {asset_id} | Delta: {delta:.2%} | Price: ${new_price:.4f}")
                                logger.info(f"🟢 Buy Signal | Asset: {asset_id} | Price: ${new_price:.4f}")
                                if place_buy_order(state, asset_id, "Spike detected"):
                                    # Only try opposite sell if we have shares to sell
                                    opposite_position = find_position_by_asset(positions_copy, opposite)
                                    if opposite_position and opposite_position.shares > KEEP_MIN_SHARES:
                                        place_sell_order(state, opposite, "Opposite trade")
                                    else:
                                        logger.info(f"🟡 Skipping opposite sell - no shares for {opposite}")
                            elif delta < 0 and not is_recently_sold(state, asset_id):
                                logger.info(f"🟨 Spike Detected | Asset: {asset_id} | Delta: {delta:.2%} | Price: ${new_price:.4f}")
                                logger.info(f"🔴 Sell Signal | Asset: {asset_id} | Price: ${new_price:.4f}")
                                # Only try to sell if we have shares
                                current_position = find_position_by_asset(positions_copy, asset_id)
                                if current_position and current_position.shares > KEEP_MIN_SHARES:
                                    if place_sell_order(state, asset_id, "Spike detected"):
                                        place_buy_order(state, opposite, "Opposite trade")
                                else:
                                    logger.info(f"🟡 Skipping sell - no shares for {asset_id}")

                    except IndexError:
                        logger.debug(f"⏳ Building price history for {asset_id}")
                        continue
                    except Exception as e:
                        logger.error(f"❌ Error processing asset {asset_id}: {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"❌ Error in detect_and_trade: {str(e)}")
            time.sleep(1)

def check_trade_exits(state: ThreadSafeState) -> None:
    last_log_time = time.time()
    
    while not state.is_shutdown():
        try:
            active_trades = state.get_active_trades()
            if active_trades:
                # Log active trades every 30 seconds instead of 5
                current_time = time.time()
                if current_time - last_log_time >= 30:
                    logger.info(f"📈 Active Trades | Count: {len(active_trades)} | Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    last_log_time = current_time
            
            for asset_id, trade in active_trades.items():
                try:
                    positions_copy = state.get_positions()
                    position = find_position_by_asset(positions_copy, asset_id)
                    if not position:
                        # Position no longer exists, remove from active trades
                        logger.warning(f"⚠️ Position not found for active trade {asset_id}, removing from tracking")
                        state.remove_active_trade(asset_id)
                        continue
                        
                    current_price = get_current_price(state, asset_id)
                    if current_price is None:
                        continue
                    
                    # Avoid division by zero
                    if position.avg_price == 0:
                        logger.warning(f"⚠️ Zero avg_price for {asset_id}, skipping exit check")
                        continue
                        
                    current_time = time.time()
                    last_traded = trade.entry_time  # entry_time is now a float timestamp
                    avg_price = position.avg_price
                    remaining_shares = position.shares
                    cash_profit = (current_price - avg_price) * remaining_shares
                    pct_profit = (current_price - avg_price) / avg_price
                    
                    # Track if we sold so we don't check multiple conditions
                    trade_exited = False

                    # Check holding time limit FIRST
                    if current_time - last_traded > HOLDING_TIME_LIMIT:
                        logger.info(f"⏰ Holding Time Limit Hit | Asset: {asset_id} | Holding Time: {current_time - last_traded:.2f} seconds | Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                        if place_sell_order(state, asset_id, "Holding time limit"):
                            state.remove_active_trade(asset_id)
                            state.set_last_trade_time(time.time())
                        trade_exited = True
                    
                    # Check take profit (use elif to avoid double-sell)
                    elif cash_profit >= CASH_PROFIT or pct_profit > PCT_PROFIT:
                        logger.info(f"🎯 Take Profit Hit | Asset: {asset_id} | Profit: ${cash_profit:.2f} ({pct_profit:.2%}) | Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                        if place_sell_order(state, asset_id, "Take profit"):
                            state.remove_active_trade(asset_id)
                            state.set_last_trade_time(time.time())
                        trade_exited = True

                    # Check stop loss (use elif to avoid double-sell)
                    elif cash_profit <= CASH_LOSS or pct_profit < PCT_LOSS:
                        logger.info(f"🔴 Stop Loss Hit | Asset: {asset_id} | Loss: ${cash_profit:.2f} ({pct_profit:.2%}) | Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                        if place_sell_order(state, asset_id, "Stop loss"):
                            state.remove_active_trade(asset_id)
                            state.set_last_trade_time(time.time())
                        trade_exited = True


                except Exception as e:
                    logger.error(f"❌ Error checking trade exit for {asset_id}: {str(e)}")
                    continue

            time.sleep(1)
            
        except Exception as e:
            logger.error(f"❌ Error in check_trade_exits: {str(e)}")
            time.sleep(1)

def get_current_price(state: ThreadSafeState, asset_id: str) -> Optional[float]:
    try:
        history = state.get_price_history(asset_id)
        if not history:
            logger.debug(f"⏳ No price history available for {asset_id}")
            return None
        return history[-1][1]
    except IndexError:
        logger.debug(f"⏳ Building price history for {asset_id}")
        return None
    except Exception as e:
        logger.error(f"❌ Error getting current price for {asset_id}: {str(e)}")
        return None

def wait_for_initialization(state: ThreadSafeState) -> bool:
    max_retries = 60
    retry_count = 0
    while retry_count < max_retries and not state.is_shutdown():
        try:
            positions = fetch_positions_with_retry()
            for event_id, sides in positions.items():
                logger.info(f"🔎 Event ID {event_id}: {len(sides)} outcomes")
                # Handle markets with 2+ outcomes
                if len(sides) >= 2:
                    ids = [s.asset for s in sides]
                    # Pair all combinations - for binary markets this pairs 0<->1
                    # For multi-outcome markets, pair each with the next
                    for i in range(len(ids)):
                        for j in range(i + 1, len(ids)):
                            state.add_asset_pair(ids[i], ids[j])
                            logger.info(f"✅ Initialized asset pair: {ids[i][:16]}... ↔ {ids[j][:16]}...")
            
            # Always return True if we successfully fetched positions (even if empty)
            logger.info(f"✅ Initialization complete. Found {len(state._initialized_assets)} tracking assets.")
            if len(state._initialized_assets) == 0:
                logger.warning("⚠️ No positions found. Bot will be idle until you open positions on Polymarket.")
            return True
                
            retry_count += 1
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"❌ Error during initialization: {str(e)}")
            retry_count += 1
            time.sleep(2)
    
    logger.warning("❌ Initialization timed out after 2 minutes.")
    return False

def sync_orphan_positions(state: ThreadSafeState) -> None:
    """Check for positions in wallet that are not tracked by bot"""
    try:
        positions_map = state.get_positions()
        active_trades = state.get_active_trades()
        
        count = 0
        for event_id, positions in positions_map.items():
            for pos in positions:
                # If position exists (> min shares) and not in active trades
                if pos.shares > KEEP_MIN_SHARES and pos.asset not in active_trades:
                    logger.info(f"🧹 Found orphan position: {pos.eventslug} ({pos.outcome}) | {pos.shares:.2f} shares")
                    
                    # Adopt it!
                    trade_id = str(uuid.uuid4())
                    
                    # Create TradeInfo
                    trade_info = TradeInfo(
                        entry_price=pos.avg_price,
                        entry_time=time.time(), # We don't know original time, assume now for holding limit
                        amount=pos.shares * pos.avg_price, # Estimate cost
                        bot_triggered=False,
                        trade_id=trade_id
                    )
                    
                    state.add_active_trade(pos.asset, trade_info)
                    
                    # Save to DB
                    if hasattr(state, 'db'):
                        record = TradeRecord(
                            id=trade_id,
                            asset_id=pos.asset,
                            entry_price=pos.avg_price,
                            entry_time=time.time(),
                            shares=pos.shares,
                            side='BUY', # Assume long
                            status='OPEN',
                            reason='Orphan adoption',
                            meta={'adopted': True}
                        )
                        state.db.add_trade(record)
                    
                    count += 1
                    notification_manager.send_message(f"🧹 **Orphan Adopted**\nAsset: `{pos.eventslug}`\nShares: {pos.shares:.2f}")

        if count > 0:
            logger.info(f"✅ Adopted {count} orphan positions")
        else:
            logger.info("✨ No orphan positions found")
            
    except Exception as e:
        logger.error(f"❌ Error syncing orphan positions: {e}")

def print_spikebot_banner() -> None:
    banner = r"""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║   ███████╗██████╗ ██╗██╗  ██╗███████╗██████╗  ██████╗ ████████╗    ║
║   ██╔════╝██╔══██╗██║██║ ██╔╝██╔════╝██╔══██╗██╔═══██╗╚══██╔══╝    ║
║   ███████╗██████╔╝██║█████╔╝ █████╗  ██████╔╝██║   ██║   ██║       ║
║   ╚════██║██╔═══╝ ██║██╔═██╗ ██╔══╝  ██╔══██╗██║   ██║   ██║       ║
║   ███████║██║     ██║██║  ██╗███████╗██████╔╝╚██████╔╝   ██║       ║
║   ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═════╝  ╚═════╝    ╚═╝       ║
║                                                                    ║
║                  🚀  P O L Y M A R K E T  B O T  🚀                ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def cleanup(state: ThreadSafeState) -> None:
    logger.info("🔄 Starting cleanup...")
    
    try:
        # Initiate shutdown
        state.shutdown()
        
        # Wait for threads to finish with timeout
        for thread in threading.enumerate():
            if thread != threading.current_thread():
                thread.join(timeout=5)
                if thread.is_alive():
                    logger.warning(f"Thread {thread.name} did not finish in time")
                    # Force terminate the thread if it's still alive
                    if hasattr(thread, '_stop'):
                        thread._stop()
        
        # Close any open connections
        try:
            # The ClobClient doesn't have a close method, so we just set it to None
            global client
            client = None
        except Exception as e:
            logger.error(f"Error closing client connection: {e}")
        
        # Wait for cleanup to complete
        if not state.wait_for_cleanup(timeout=10):
            logger.warning("Cleanup did not complete in time")
        
        logger.info("✅ Cleanup complete")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise

def signal_handler(signum: int, frame: Any, state: ThreadSafeState) -> None:
    logger.info(f"Received signal {signum}. Initiating shutdown...")
    cleanup(state)
    sys.exit(0)

def main() -> None:
    # Declare globals we might need to update
    global CONFIG_VALID, YOUR_PROXY_WALLET, BOT_TRADER_ADDRESS, PRIVATE_KEY
    
    state = None
    thread_manager = None
    try:
        print_spikebot_banner()
        
        # Initialize state for dashboard
        state = ThreadSafeState()
        
        # Start dashboard first so user can configure
        logger.info("📊 Starting dashboard server on http://localhost:5000")
        start_time = time.time()
        dashboard_thread = start_dashboard(state, start_time)
        logger.info("✅ Dashboard started successfully")
        
        # Check configuration
        if not CONFIG_VALID:
            logger.warning("⚠️ Configuration incomplete. Please configure via dashboard at http://localhost:5000/settings")
            logger.warning(f"Missing/Invalid: {CONFIG_ERROR}")
            
            # Wait for configuration to be valid
            while not CONFIG_VALID:
                time.sleep(2)
                # Reload env to check if user saved settings
                from dotenv import load_dotenv
                load_dotenv(override=True)
                
                # Re-validate
                valid, error = validate_config()
                if valid:
                    logger.info("✅ Configuration updated and valid! Starting bot...")
                    # Update global variables
                    CONFIG_VALID = True
                    YOUR_PROXY_WALLET = get_checksum_address("YOUR_PROXY_WALLET")
                    BOT_TRADER_ADDRESS = get_checksum_address("BOT_TRADER_ADDRESS")
                    PRIVATE_KEY = os.getenv("PK")
                    notification_manager.reload_config()
                    break
        
        # Initialize CLOB client
        global client
        try:
            client = initialize_clob_client()
            logger.info("✅ CLOB Client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Critical error initializing CLOB client: {e}")
            return

        thread_manager = ThreadManager(state)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, state))
        signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, state))
        
        # Initialize
        spinner = Halo(text="Initializing bot components...", spinner="dots")
        spinner.start()
        time.sleep(2)
        
        if not wait_for_initialization(state):
            spinner.fail("❌ Failed to initialize. Check your network or wallet settings.")
            # Don't crash, just wait for user to fix settings
            while not state.is_shutdown():
                time.sleep(1)
            return
        
        spinner.succeed("Initialized successfully")
        logger.info(f"🚀 Spike-detection bot started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        notification_manager.send_startup()
        
        # Check for orphans
        sync_orphan_positions(state)
        
        # Start price update thread first and wait for initial data
        logger.info("🔄 Starting price update thread...")
        thread_manager.start_thread("price_update", update_price_history)
        
        # Wait for initial price data
        logger.info("⏳ Waiting for initial price data...")
        initial_data_wait = 0
        while initial_data_wait < 30:  # Wait up to 30 seconds for initial data
            if any(state.get_price_history(asset_id) for asset_id in state._price_history.keys()):
                logger.info("✅ Initial price data received")
                break
            time.sleep(1)
            initial_data_wait += 1
            if initial_data_wait % 5 == 0:
                logger.info(f"⏳ Still waiting for initial price data... ({initial_data_wait}/30 seconds)")
        
        if initial_data_wait >= 30:
            logger.warning("⚠️ No initial price data received after 30 seconds")
        
        # Start trading threads
        logger.info("🔄 Starting trading threads...")
        thread_manager.start_thread("detect_trade", detect_and_trade)
        thread_manager.start_thread("check_exits", check_trade_exits)
        
        last_refresh_time = time.time()
        refresh_interval = REFRESH_INTERVAL
        last_status_time = time.time()
        
        # Main loop
        while not state.is_shutdown():
            try:
                current_time = time.time()
                
                # Daily reset at midnight UTC
                # if current_time - last_daily_reset >= 86400:  # 24 hours
                #     logger.info("🔄 Performing daily reset...")
                #     state.reset_daily_pnl()
                #     last_daily_reset = current_time
                
                # Log status every 30 seconds
                if current_time - last_status_time >= 30:
                    active_threads = sum(1 for t in thread_manager.threads.values() if t.is_alive())
                    logger.info(f"📊 Bot Status | Active Threads: {active_threads}/3 | Price Updates: {len(state._price_history)}")
                    last_status_time = current_time
                
                # Refresh API credentials
                if current_time - last_refresh_time > refresh_interval:
                    if refresh_api_credentials():
                        last_refresh_time = current_time
                    else:
                        logger.warning("⚠️ Failed to refresh API credentials. Will retry in 5 minutes.")
                        time.sleep(300)
                        continue
                
                # Check if any threads have died - use proper name-to-function mapping
                thread_function_map = {
                    "price_update": update_price_history,
                    "detect_trade": detect_and_trade,
                    "check_exits": check_trade_exits
                }
                for name, thread in thread_manager.threads.items():
                    if not thread.is_alive():
                        logger.warning(f"⚠️ Thread {name} has died. Restarting...")
                        if name in thread_function_map:
                            thread_manager.start_thread(name, thread_function_map[name])
                        else:
                            logger.error(f"❌ Unknown thread name: {name}, cannot restart")
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("👋 Shutting down gracefully...")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
    finally:
        if thread_manager:
            thread_manager.stop()
        if state:
            cleanup(state)

if __name__ == "__main__":
    main()