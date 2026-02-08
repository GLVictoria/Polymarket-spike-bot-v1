
import os
import sys
from dotenv import load_dotenv
from web3 import Web3
from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON

# Load environment variables
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', '.env')
if os.path.isfile(config_path):
    load_dotenv(config_path)
else:
    load_dotenv()

pk = os.getenv("PK")
if not pk:
    print("❌ Error: PK (Private Key) not found in .env or environment variables.")
    print("Please set PK in your .env file.")
    sys.exit(1)

# derive EOA
try:
    w3 = Web3()
    account = w3.eth.account.from_key(pk)
    eoa_address = account.address
except Exception as e:
    print(f"❌ Error deriving address from PK: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("🔑  YOUR WALLET ADDRESSES")
print("="*60)

print(f"\n1. SIGNER ACCOUNT (EOA)")
print(f"   Address: {eoa_address}")
print(f"   (This is the wallet you exported the Private Key from)")
print(f"   Balance needed: MATIC (for gas fees)")

# derive/check Proxy
print(f"\n2. POLYMARKET PROXY WALLET")
print("   Connecting to Polymarket API to find your proxy wallet...")

try:
    # Initialize client with EOA as signer
    # We try to derive the API credentials which should include the proxy address
    client = ClobClient(
        host="https://clob.polymarket.com",
        key=pk,
        chain_id=137
    )
    
    # Check if we can get credentials
    try:
        # standard way to get credentials
        creds = client.create_or_derive_api_creds()
        print(f"   ✅ API Credentials Retrieved Successfully")
        
        # The proxy address is usually associated with these credentials
        # Or often the 'funder' needs to be set to the proxy address for trading
        # If the user has a proxy deployed, the API might tell us.
        
        # Another check via user object if possible?
        # client.get_account() might return account details including proxy address
        
        # Let's try to infer if we can get the proxy address from the creds
        # Usually creds have 'api_key', 'api_secret', 'api_passphrase'
        
        # If we look at how the main bot initializes:
        # funder=YOUR_PROXY_WALLET
        
        # Let's try to see if the client has a method to get the proxy
        # If not, we fall back to manual instructions
        
        print("\n   ℹ️  INSTRUCTIONS:")
        print("   If you don't know your Proxy Wallet address:")
        print("   1. Go to https://polymarket.com")
        print("   2. Click on your profile/wallet icon")
        print("   3. Click 'Copy Address'")
        print("   4. Paste that address into your .env file as YOUR_PROXY_WALLET")
        print("   (It should be different from your Signer Address if using a Proxy)")
        
    except Exception as e:
        print(f"   ⚠️  Could not automatically derive API credentials: {e}")

except Exception as e:
    print(f"   ❌ Error connecting to Polymarket: {e}")

print("\n" + "="*60)
print("📝  NEXT STEPS")
print("1. Copy the 'Proxy Wallet' address from Polymarket website")
print("2. Paste it into your .env file as YOUR_PROXY_WALLET")
print("3. Ensure 'BOT_TRADER_ADDRESS' in .env matches your Signer Account (EOA)")
print("4. Ensure 'PK' matches the Private Key for that Signer Account")
print("="*60 + "\n")
