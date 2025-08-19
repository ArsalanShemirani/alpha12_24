#!/usr/bin/env python3
"""
Load environment variables from .env file for alpha12_24
"""

import os
import sys
from pathlib import Path

def load_env():
    """Load environment variables from .env file"""
    env_file = Path(".env")
    
    if not env_file.exists():
        print("⚠️  .env file not found!")
        print("Please create a .env file with your Telegram bot credentials.")
        return False
    
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        # Expand $(pwd) if present
                        if '$(pwd)' in value:
                            import subprocess
                            pwd_result = subprocess.run(['pwd'], capture_output=True, text=True)
                            if pwd_result.returncode == 0:
                                value = value.replace('$(pwd)', pwd_result.stdout.strip())
                        os.environ[key.strip()] = value.strip()
        
        print("✅ Environment variables loaded successfully!")
        
        # Verify Telegram credentials
        tg_token = os.getenv("TG_BOT_TOKEN") or os.getenv("TG_BOT")
        tg_chat = os.getenv("TG_CHAT_ID") or os.getenv("TG_CHAT")
        
        if tg_token and tg_chat:
            print("✅ Telegram bot credentials loaded:")
            print(f"   Bot Token: {tg_token[:10]}...")
            print(f"   Chat ID: {tg_chat}")
            return True
        else:
            print("❌ Telegram bot credentials not found!")
            return False
            
    except Exception as e:
        print(f"❌ Error loading environment variables: {e}")
        return False

if __name__ == "__main__":
    success = load_env()
    if success:
        print("\n🚀 Environment ready! You can now run:")
        print("   Dashboard: streamlit run src/dashboard/app.py")
        print("   Tracker: python src/daemon/tracker.py")
    else:
        sys.exit(1)
