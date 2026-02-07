
import requests
import os
import logging
from typing import Optional

logger = logging.getLogger('polymarket_bot')

class NotificationManager:
    def __init__(self):
        self.telegram_token: Optional[str] = None
        self.telegram_chat_id: Optional[str] = None
        self.discord_webhook: Optional[str] = None
        self.reload_config()

    def reload_config(self):
        """Reload configuration from environment variables"""
        self.telegram_token = os.getenv("TELEGRAM_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.discord_webhook = os.getenv("DISCORD_WEBHOOK_URL")
        
        if self.telegram_token:
            logger.info("✅ Telegram notifications enabled")
        if self.discord_webhook:
            logger.info("✅ Discord notifications enabled")

    def send_message(self, message: str) -> None:
        """Send message to all configured channels"""
        self._send_telegram(message)
        self._send_discord(message)

    def _send_telegram(self, message: str) -> None:
        if not self.telegram_token or not self.telegram_chat_id:
            return
            
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"❌ Failed to send Telegram message: {e}")

    def _send_discord(self, message: str) -> None:
        if not self.discord_webhook:
            return
            
        try:
            payload = {"content": message}
            requests.post(self.discord_webhook, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"❌ Failed to send Discord message: {e}")

    def send_trade_alert(self, side: str, asset: str, price: float, amount: float, reason: str) -> None:
        """Send formatted trade alert"""
        emoji = "🟢" if side.upper() == "BUY" else "🔴"
        # Truncate asset ID for readability
        short_asset = asset[:8] + "..." if len(asset) > 10 else asset
        
        msg = (
            f"{emoji} *{side.upper()} ALERT*\n\n"
            f"🎯 Asset: `{short_asset}`\n"
            f"💰 Price: `${price:.4f}`\n"
            f"💵 Value: `${amount:.2f}`\n"
            f"📋 Reason: {reason}"
        )
        self.send_message(msg)

    def send_error(self, error_msg: str) -> None:
        """Send error alert"""
        self.send_message(f"❌ *ERROR ALERT*\n\n{error_msg}")

    def send_startup(self) -> None:
        """Send startup message"""
        self.send_message("🚀 *PolySpike Bot Started*\n\nMonitoring markets for spikes...")

# Global instance
notification_manager = NotificationManager()
