# PolySpike Trader

A real-time trading bot for Polymarket that detects price spikes and executes trades automatically.

## Features

- 📈 **Spike Detection** - Monitors price movements and detects rapid changes
- ⚡ **Automatic Trading** - Places buy/sell orders based on spike direction
- 🎯 **Take Profit / Stop Loss** - Configurable exit conditions
- 📊 **Real-time Dashboard** - Monitor trades, positions, and logs via web UI
- ⚙️ **Easy Configuration** - Configure wallet and settings directly in the dashboard
- 🐳 **Docker Ready** - Simple one-command deployment
- 🔒 **Secure** - Private keys are handled securely and never exposed in logs

## Prerequisites

Before you start, make sure you have Docker installed:

1. **Download Docker Desktop**: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
2. **Install & Run**: Follow the installer instructions and ensure Docker Desktop is running in the background.

## Quick Start (Recommended)

The easiest way to run the bot is using Docker. This ensures a consistent environment and automatic restarts.

### 1. Start the Bot
Run the following command in the project directory:

```bash
docker-compose up -d
```

### 2. Configure via Dashboard
Open your browser to **http://localhost:5000/settings**

1. Enter your **Private Key**, **Proxy Wallet**, and **Trader Address**
2. Adjust trading parameters (optional)
3. Click **Save Settings**

The bot will automatically detect the configuration and start trading! 🚀

### 3. Monitor
Go to **http://localhost:5000** to see:
- Bot status & uptime
- Active trades with live PnL
- Position overview
- Price monitor
- Trade history
- Live streaming logs

---

## Manual Installation (Python)

If you prefer to run without Docker:

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Bot**
   ```bash
   python main.py
   ```

3. **Configure**
   - Open **http://localhost:5000/settings** to configure the bot.
   - Or manually copy `example.env` to `.env` and edit it.

## Configuration

You can configure these settings via the Dashboard or `.env` file:

| Setting | Description | Default |
|---------|-------------|---------|
| `trade_unit` | Trade size in USDC | 3 |
| `slippage_tolerance` | Max slippage allowed | 0.06 (6%) |
| `spike_threshold` | Price change to trigger trade | 0.01 (1%) |
| `pct_profit` | Take profit percentage | 0.03 (3%) |
| `pct_loss` | Stop loss percentage | -0.025 (-2.5%) |
| `holding_time_limit` | Max time to hold a position | 60s |
| `cooldown_period` | Wait time between trades | 120s |

## dashboard

The dashboard runs on port `5000` by default.

- **Home**: http://localhost:5000
- **Settings**: http://localhost:5000/settings
- **API**: http://localhost:5000/api/status

## Architecture

```mermaid
graph TD
    A[Price Feed] -->|Updates| B(Spike Detector)
    B -->|Signal| C{Trade Engine}
    C -->|Buy/Sell| D[Polymarket API]
    E[Exit Monitor] -->|Check PnL| C
    
    subgraph Dashboard
    F[Web Receiver] -->|WebSocket| G[Browser UI]
    H[Settings API] -->|Write| I[.env File]
    end
    
    C -->|Log/Update| F
```

## Troubleshooting

- **Bot not trading?** Check the "Live Logs" on the dashboard for errors.
- **Dashboard 500 Error?** Ensure the `.env` file is writable or create it via the Settings page.
- **Docker issues?** Run `docker-compose logs -f` to see container output.

## License

MIT
