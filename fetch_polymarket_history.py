
import os
import requests
import json
from dotenv import load_dotenv

# Load env
# Check standard paths
paths = [
    "/app/config/.env",
    "config/.env",
    ".env"
]
for p in paths:
    if os.path.exists(p):
        load_dotenv(p)
        break

proxy_wallet = os.getenv("YOUR_PROXY_WALLET")
if not proxy_wallet:
    print("❌ YOUR_PROXY_WALLET not found in environment")
    exit(1)

print(f"🔍 Fetching history for Proxy Wallet: {proxy_wallet}")

def fetch_and_print(url, name):
    print(f"\n📡 Fetching {name}...")
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                print(f"✅ Found {len(data)} records!")
                print("--- Recent Activity ---")
                for item in data[:5]: # Show last 5
                    print(json.dumps(item, indent=2))
            elif isinstance(data, list) and len(data) == 0:
                print("⚠️ No records found (Empty list)")
            else:
                print(f"⚠️ Unexpected response format: {type(data)}")
                print(data)
        else:
            print(f"❌ Error {r.status_code}: {r.text}")
    except Exception as e:
        print(f"❌ Failed: {e}")

# Try Activity endpoint
fetch_and_print(f"https://data-api.polymarket.com/activity?user={proxy_wallet}&limit=20", "User Activity")

# Try Trades endpoint (as maker)
fetch_and_print(f"https://data-api.polymarket.com/trades?maker_address={proxy_wallet}&limit=20", "Trades (Maker)")

# Try Trades endpoint (as taker)
fetch_and_print(f"https://data-api.polymarket.com/trades?taker_address={proxy_wallet}&limit=20", "Trades (Taker)")

print("\n(Note: If all are empty, this wallet has likely never traded on Polymarket via this proxy)")
