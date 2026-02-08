"""
Polymarket Spike Bot - Real-time Dashboard
Provides a web interface to monitor bot activities
"""

import json
import os
import logging
import threading
import time
from datetime import datetime
from collections import deque
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from typing import Optional, Dict, Any, List
import requests

# Dashboard configuration
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 5000
MAX_LOG_BUFFER = 200

app = Flask(__name__)
app.config['SECRET_KEY'] = 'polymarket-spike-bot-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global reference to bot state (set from main.py)
_bot_state = None
_bot_start_time = None
_log_buffer = deque(maxlen=MAX_LOG_BUFFER)
_trade_history = deque(maxlen=50)


class DashboardLogHandler(logging.Handler):
    """Custom log handler that sends logs to the dashboard"""
    
    def emit(self, record):
        try:
            log_entry = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'level': record.levelname,
                'message': self.format(record),
                'thread': record.threadName
            }
            _log_buffer.append(log_entry)
            # Emit to connected clients
            socketio.emit('log', log_entry, namespace='/dashboard')
        except Exception:
            pass


def set_bot_state(state, start_time=None):
    """Set the bot state reference for the dashboard"""
    global _bot_state, _bot_start_time
    _bot_state = state
    _bot_start_time = start_time or time.time()


def add_trade_to_history(trade_type: str, asset: str, price: float, amount: float, reason: str):
    """Add a trade to the history for dashboard display"""
    trade = {
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'type': trade_type,
        'asset': asset[:16] + '...' if len(asset) > 16 else asset,
        'price': price,
        'amount': amount,
        'reason': reason
    }
    _trade_history.append(trade)
    socketio.emit('trade', trade, namespace='/dashboard')


def get_dashboard_logger() -> logging.Handler:
    """Get the dashboard log handler to add to the main logger"""
    handler = DashboardLogHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(message)s'))
    return handler


# Routes
@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/favicon.ico')
def favicon():
    return '', 204


@app.route('/api/status')
def api_status():
    """Get current bot status"""
    if _bot_state is None:
        return jsonify({'error': 'Bot not initialized'}), 503
    
    uptime = time.time() - _bot_start_time if _bot_start_time else 0
    hours, remainder = divmod(int(uptime), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    active_trades = _bot_state.get_active_trades()
    positions = _bot_state.get_positions()
    
    return jsonify({
        'paused': _bot_state.is_paused(),
        'running': not _bot_state.is_shutdown(),
        'uptime': f'{hours:02d}:{minutes:02d}:{seconds:02d}',
        'active_trades': len(active_trades),
        'tracked_assets': len(_bot_state._price_history),
        'positions_count': sum(len(p) for p in positions.values()),
        'balance': _bot_state.get_balance(),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })


@app.route('/api/trades')
def api_trades():
    """Get active trades"""
    if _bot_state is None:
        return jsonify({'error': 'Bot not initialized'}), 503
    
    active_trades = _bot_state.get_active_trades()
    positions = _bot_state.get_positions()
    
    trades_data = []
    for asset_id, trade in active_trades.items():
        # Find position for this asset
        position = None
        for pos_list in positions.values():
            for p in pos_list:
                if p.asset == asset_id:
                    position = p
                    break
        
        current_price = 0
        pnl = 0
        pnl_pct = 0
        
        if position:
            current_price = position.current_price
            if position.avg_price > 0:
                pnl = (current_price - position.avg_price) * position.shares
                pnl_pct = ((current_price - position.avg_price) / position.avg_price) * 100
        
        trades_data.append({
            'asset': asset_id[:16] + '...' if len(asset_id) > 16 else asset_id,
            'entry_price': trade.entry_price,
            'current_price': current_price,
            'amount': trade.amount,
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct, 2),
            'holding_time': int(time.time() - trade.entry_time)
        })
    
    return jsonify(trades_data)


@app.route('/api/positions')
def api_positions():
    """Get all positions"""
    if _bot_state is None:
        return jsonify({'error': 'Bot not initialized'}), 503
    
    positions = _bot_state.get_positions()
    positions_data = []
    
    for event_id, pos_list in positions.items():
        for p in pos_list:
            positions_data.append({
                'event': p.eventslug[:20] + '...' if len(p.eventslug) > 20 else p.eventslug,
                'outcome': p.outcome,
                'shares': round(p.shares, 2),
                'avg_price': round(p.avg_price, 4),
                'current_price': round(p.current_price, 4),
                'pnl': round(p.pnl, 2),
                'pnl_pct': round(p.percent_pnl * 100, 2)
            })
    
    return jsonify(positions_data)


@app.route('/api/prices')
def api_prices():
    """Get current prices for tracked assets"""
    if _bot_state is None:
        return jsonify({'error': 'Bot not initialized'}), 503
    
    prices_data = []
    positions = _bot_state.get_positions()
    
    for event_id, pos_list in positions.items():
        for p in pos_list:
            history = _bot_state.get_price_history(p.asset)
            price_change = 0
            if len(history) >= 2:
                old_price = history[-min(5, len(history))][1]
                if old_price > 0:
                    price_change = ((p.current_price - old_price) / old_price) * 100
            
            prices_data.append({
                'asset': p.asset[:16] + '...',
                'event': p.eventslug[:15] + '...' if len(p.eventslug) > 15 else p.eventslug,
                'outcome': p.outcome,
                'price': round(p.current_price, 4),
                'change': round(price_change, 2)
            })
    
    return jsonify(prices_data)


@app.route('/api/history')
def api_history():
    """Get trade history"""
    return jsonify(list(_trade_history))



@app.route('/api/portfolio-history')
def api_portfolio_history():
    """Get full portfolio history from Polymarket API"""
    try:
        # Get proxy wallet from settings
        settings = read_env_file()
        proxy_wallet = settings.get('YOUR_PROXY_WALLET')
        
        if not proxy_wallet:
            return jsonify({'error': 'Proxy wallet not configured'}), 400
            
        # Fetch activity from Polymarket API
        # We use the activity endpoint as it gives a good overview of actions
        url = f"https://data-api.polymarket.com/activity?user={proxy_wallet}&limit=50"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return jsonify(data)
        else:
            return jsonify({'error': f'API Error: {response.status_code}'}), 502
            
    except Exception as e:
        print(f"Error fetching portfolio history: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/api/watched')
def api_get_watched():
    if _bot_state is None:
        return jsonify({'error': 'Bot not initialized'}), 503
    return jsonify(_bot_state.get_watched_markets())

@app.route('/api/watch', methods=['POST'])
def api_watch_market():
    if _bot_state is None:
        return jsonify({'error': 'Bot not initialized'}), 503
        
    data = request.get_json()
    if not data or 'market_id' not in data:
        return jsonify({'error': 'Missing market_id'}), 400
        
    market_id = data['market_id'].strip()
    if not market_id or not market_id.startswith('0x'): # Basic validation
         # Try to extract from URL if pasted
        import re
        match = re.search(r'0x[a-fA-F0-9]{40,}', market_id) # Token ID is usually long hex
        if match:
            market_id = match.group(0)
        else:
            # Maybe it's a condition ID, also hex
            pass
            
    if _bot_state.add_watched_market(market_id):
        return jsonify({'success': True, 'message': f'Now watching {market_id}'})
    else:
        return jsonify({'success': False, 'message': 'Already watching this market'})

@app.route('/api/unwatch', methods=['POST'])
def api_unwatch_market():
    if _bot_state is None:
        return jsonify({'error': 'Bot not initialized'}), 503
        
    data = request.get_json()
    if not data or 'market_id' not in data:
        return jsonify({'error': 'Missing market_id'}), 400
        
    if _bot_state.remove_watched_market(data['market_id']):
        return jsonify({'success': True, 'message': 'Stopped watching market'})
    else:
        return jsonify({'success': False, 'message': 'Market was not being watched'})

@app.route('/api/logs')
def api_logs():
    """Get recent logs"""
    return jsonify(list(_log_buffer))


# Settings configuration
# Use config directory for Docker volume mount (prevents directory creation issue)
import os
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
ENV_FILE_PATH = os.path.join(CONFIG_DIR, '.env') if os.path.isdir(CONFIG_DIR) else '.env'
SETTINGS_SCHEMA = {
    'wallet': {
        'PK': {'label': 'Private Key', 'type': 'password', 'required': True},
        'YOUR_PROXY_WALLET': {'label': 'Proxy Wallet', 'type': 'text', 'required': True},
        'BOT_TRADER_ADDRESS': {'label': 'Trader Address', 'type': 'text', 'required': True},
    },
    'network': {
        'USDC_CONTRACT_ADDRESS': {'label': 'USDC Contract (Polygon)', 'type': 'text', 'default': '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174', 'required': True},
        'POLYMARKET_SETTLEMENT_CONTRACT': {'label': 'Settlement Contract', 'type': 'text', 'default': '0x56C79347e95530c01A2FC76E732f9566dA16E113', 'required': True},
    },
    'trading': {
        'trade_unit': {'label': 'Trade Size (USDC)', 'type': 'number', 'default': '3'},
        'slippage_tolerance': {'label': 'Slippage Tolerance', 'type': 'number', 'default': '0.06'},
        'spike_threshold': {'label': 'Spike Threshold', 'type': 'number', 'default': '0.01'},
    },
    'risk': {
        'pct_profit': {'label': 'Take Profit %', 'type': 'number', 'default': '0.03'},
        'pct_loss': {'label': 'Stop Loss %', 'type': 'number', 'default': '-0.025'},
        'cash_profit': {'label': 'Take Profit ($)', 'type': 'number', 'default': '3'},
        'cash_loss': {'label': 'Stop Loss ($)', 'type': 'number', 'default': '-3'},
        'holding_time_limit': {'label': 'Max Hold Time (sec)', 'type': 'number', 'default': '60'},
        'max_daily_loss': {'label': 'Max Daily Loss ($)', 'type': 'number', 'default': '-10'},
    },
    'notifications': {
        'TELEGRAM_TOKEN': {'label': 'Telegram Bot Token', 'type': 'password', 'required': False},
        'TELEGRAM_CHAT_ID': {'label': 'Telegram Chat ID', 'type': 'text', 'required': False},
        'DISCORD_WEBHOOK_URL': {'label': 'Discord Webhook URL', 'type': 'password', 'required': False},
    },
    'advanced': {
        'cooldown_period': {'label': 'Cooldown (sec)', 'type': 'number', 'default': '120'},
        'sold_position_time': {'label': 'Position Cooldown (sec)', 'type': 'number', 'default': '1800'},
        'price_history_size': {'label': 'Price History Size', 'type': 'number', 'default': '120'},
        'keep_min_shares': {'label': 'Min Shares to Keep', 'type': 'number', 'default': '1'},
        'max_concurrent_trades': {'label': 'Max Concurrent Trades', 'type': 'number', 'default': '3'},
        'min_liquidity_requirement': {'label': 'Min Liquidity (USDC)', 'type': 'number', 'default': '10'},
    }
}


def read_env_file() -> Dict[str, str]:
    """Read .env file and return as dictionary"""
    import os
    
    # Try multiple possible locations (config dir first for Docker)
    possible_paths = [
        "/app/config/.env",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', '.env'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'),
        os.path.abspath('.env'),
        "/app/.env"
    ]
    
    settings = {}
    env_path = None
    
    for path in possible_paths:
        if os.path.isfile(path):  # Check it's a file, not directory
            env_path = path
            break
            
    if not env_path:
        print(f"Warning: .env file not found in {possible_paths}")
        return settings
    
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    settings[key.strip()] = value.strip()
    except Exception as e:
        print(f"Error reading .env file: {e}")
        
    return settings


def write_env_file(settings: Dict[str, str]) -> bool:
    """Write settings to .env file"""
    import os
    
    # Try multiple possible locations (config dir first for Docker)
    possible_paths = [
        "/app/config/.env",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', '.env'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'),
        os.path.abspath('.env'),
        "/app/.env"
    ]
    
    env_path = None
    for path in possible_paths:
        if os.path.isfile(path):  # Check it's a file, not directory
            env_path = path
            break
    
    # If not found, create in config directory (preferred) or app directory
    if not env_path:
        config_dir = "/app/config" if os.path.isdir("/app/config") else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
        if os.path.isdir(config_dir):
            env_path = os.path.join(config_dir, '.env')
        else:
            env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    
    print(f"Saving settings to: {env_path}")
    
    try:
        # Read existing file to preserve comments and order
        lines = []
        found_keys = set()
        
        if os.path.isfile(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    stripped = line.strip()
                    if stripped and not stripped.startswith('#') and '=' in stripped:
                        key = stripped.split('=', 1)[0].strip()
                        if key in settings:
                            lines.append(f"{key}={settings[key]}\n")
                            found_keys.add(key)
                        else:
                            lines.append(line)
                    else:
                        lines.append(line)
        
        # Add any new keys that weren't in the original file
        for key, value in settings.items():
            if key not in found_keys:
                lines.append(f"{key}={value}\n")
        
        with open(env_path, 'w') as f:
            f.writelines(lines)
            
        print("Successfully wrote .env file")
        return True
    except Exception as e:
        print(f"Error writing .env file: {e}")
        return False


def mask_sensitive_value(key: str, value: str) -> str:
    """Mask sensitive values like private keys"""
    if key == 'PK' and value:
        if len(value) > 8:
            return '•' * (len(value) - 4) + value[-4:]
        return '•' * len(value)
    return value


@app.route('/settings')
def settings_page():
    """Render settings page"""
    return render_template('settings.html')


@app.route('/api/settings')
def api_get_settings():
    """Get current settings"""
    try:
        current_settings = read_env_file()
        
        # Build response with schema and current values
        response = {}
        for category, fields in SETTINGS_SCHEMA.items():
            response[category] = {}
            for key, config in fields.items():
                value = current_settings.get(key, config.get('default', ''))
                response[category][key] = {
                    'label': config['label'],
                    'type': config['type'],
                    'value': mask_sensitive_value(key, value),
                    'required': config.get('required', False),
                    'hasValue': bool(value)
                }
        
        return jsonify(response)
    except Exception as e:
        print(f"Error in api_get_settings: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/settings', methods=['POST'])
def api_save_settings():
    """Save settings to .env file"""
    from flask import request
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Read current settings
    current_settings = read_env_file()
    
    # Update with new values (skip masked values)
    for key, value in data.items():
        # Don't overwrite PK if it's masked (hasn't been changed)
        if key == 'PK' and value and value.startswith('•'):
            continue
        if value is not None and value != '':
            current_settings[key] = str(value)
    
    # Write to file
    if write_env_file(current_settings):
        return jsonify({'success': True, 'message': 'Settings saved. Restart bot to apply changes.'})
    else:
        return jsonify({'error': 'Failed to save settings'}), 500


@app.route('/api/settings/schema')
def api_settings_schema():
    """Get settings schema"""
    return jsonify(SETTINGS_SCHEMA)


# WebSocket events
@socketio.on('connect', namespace='/dashboard')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'status': 'ok'})


def start_dashboard(state, start_time=None):
    """Start the dashboard server in a background thread"""
    set_bot_state(state, start_time)
    
    def run_server():
        socketio.run(app, host=DASHBOARD_HOST, port=DASHBOARD_PORT, 
                    debug=False, use_reloader=False, log_output=False,
                    allow_unsafe_werkzeug=True)
    
    thread = threading.Thread(target=run_server, daemon=True, name="Dashboard")
    thread.start()
    return thread


@app.route('/api/restart', methods=['POST'])
def api_restart():
    """Restart the bot"""
    def restart_process():
        # Give time for response to be sent
        time.sleep(1)
        # Force exit, Docker will restart it
        os._exit(1)
        
    threading.Thread(target=restart_process).start()
    return jsonify({'success': True, 'message': 'Bot restarting...'})

@app.route('/api/pause', methods=['POST'])
def api_pause():
    """Toggle bot pause state"""
    if _bot_state is None:
        return jsonify({'error': 'Bot not initialized'}), 503
    
    new_state = _bot_state.toggle_pause()
    status = "PAUSED" if new_state else "RUNNING"
    return jsonify({'success': True, 'paused': new_state, 'message': f'Bot is now {status}'})


# Price update emitter (call this from main bot)
def emit_price_update(data: Dict[str, Any]):
    """Emit price update to connected clients"""
    socketio.emit('price_update', data, namespace='/dashboard')
