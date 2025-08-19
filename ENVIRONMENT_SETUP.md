# Environment Setup Guide for alpha12_24

## 📱 Telegram Bot Setup

### 1. Create a Telegram Bot
1. Open Telegram and search for `@BotFather`
2. Send `/newbot` command
3. Follow the instructions to create your bot
4. Save the bot token (format: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

### 2. Get Your Chat ID
1. Send a message to your bot
2. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
3. Find your chat ID in the response (it's a number)

### 3. Configure Environment Variables

The system automatically loads environment variables from a `.env` file. You can create this file manually or use the provided scripts.

#### Option A: Manual Setup
Create a `.env` file in the project root:

```bash
# Telegram Bot Configuration
TG_BOT_TOKEN=8119234697:AAE7dGn707CEXZo0hyzSHpzCkQtIklEhDkE
TG_CHAT_ID=4873132631

# Alternative environment variable names (for compatibility)
TG_BOT=8119234697:AAE7dGn707CEXZo0hyzSHpzCkQtIklEhDkE
TG_CHAT=4873132631

# Alpha12_24 Configuration
ALPHA12_SYMBOL=BTCUSDT
ALPHA12_INTERVAL=5m
ALPHA12_SLEEP=15
MAX_SETUPS_PER_DAY=2
PYTHONPATH=$(pwd)

# Optional: Runs directory (default: runs)
# RUNS_DIR=runs
```

#### Option B: Using the Scripts

**Shell Script (Linux/macOS):**
```bash
# Make executable (if needed)
chmod +x load_env.sh

# Load environment variables
source load_env.sh
```

**Python Script:**
```bash
python load_env.py
```

## 🔧 Environment Variables Explained

### Telegram Configuration
- `TG_BOT_TOKEN`: Your Telegram bot token
- `TG_CHAT_ID`: Your Telegram chat ID for receiving alerts

### Alpha12_24 Configuration
- `ALPHA12_SYMBOL`: Default trading symbol (default: BTCUSDT)
- `ALPHA12_INTERVAL`: Default time interval (default: 5m)
- `ALPHA12_SLEEP`: Tracker sleep interval in seconds (default: 15)
- `MAX_SETUPS_PER_DAY`: Maximum number of setups per day per asset/interval (default: 2)
- `PYTHONPATH`: Python path for module imports (default: $(pwd))

### Optional Configuration
- `RUNS_DIR`: Directory for storing runs data (default: runs)

## 🚀 Running with Telegram Alerts

### Dashboard with Alerts
```bash
# Load environment and run dashboard
source load_env.sh && streamlit run src/dashboard/app.py
```

### Tracker with Alerts
```bash
# Load environment and run tracker
source load_env.sh && python src/daemon/tracker.py
```

### Test Telegram Alerts
```bash
# Test if alerts are working
python test_setup_lifecycle.py
```

## 📋 Telegram Alert Types

The system sends the following types of alerts:

### 1. Setup Creation
```
🎯 New Setup Created
BTCUSDT 5m LONG
Entry: 50000.00 | Stop: 49500.00 | Target: 51000.00
Confidence: 75%
```

### 2. Setup Triggered
```
🎯 Setup TRIGGERED
BTCUSDT 5m LONG
Entry: 50000.00 → 50010.00
Stop: 49500.00 | Target: 51000.00
```

### 3. Target Hit
```
✅ Setup TARGET
BTCUSDT 5m LONG
Entry: 50000.00 → Exit: 51000.00
PnL: 2.00%
```

### 4. Stop Loss Hit
```
❌ Setup STOP
BTCUSDT 5m LONG
Entry: 50000.00 → Exit: 49500.00
PnL: -1.00%
```

### 5. Setup Timeout
```
⏰ Setup TIMEOUT
BTCUSDT 5m LONG
Entry: 50000.00 → Exit: 50005.00
PnL: 0.01%
```

### 6. Setup Cancelled
```
❌ Setup CANCELLED
BTCUSDT 5m LONG
Setup ID: test_1234567890
```

## 🔧 Troubleshooting

### Telegram Alerts Not Working?

1. **Check Bot Token:**
   ```bash
   echo $TG_BOT_TOKEN
   ```

2. **Check Chat ID:**
   ```bash
   echo $TG_CHAT_ID
   ```

3. **Test Bot Manually:**
   ```bash
   curl "https://api.telegram.org/bot$TG_BOT_TOKEN/sendMessage" \
        -d "chat_id=$TG_CHAT_ID" \
        -d "text=Test message"
   ```

4. **Check Bot Permissions:**
   - Make sure your bot is not blocked
   - Send `/start` to your bot if you haven't already

### Environment Variables Not Loading?

1. **Check .env file exists:**
   ```bash
   ls -la .env
   ```

2. **Check file format:**
   ```bash
   cat .env
   ```

3. **Manual load:**
   ```bash
   source load_env.sh
   ```

## 📁 File Structure

```
alpha12_24/
├── .env                    # Environment variables (create this)
├── env_template.txt        # Template for .env file
├── load_env.sh            # Shell script to load environment
├── load_env.py            # Python script to load environment
├── test_setup_lifecycle.py # Test script for alerts
└── ENVIRONMENT_SETUP.md   # This file
```

## ✅ Verification

To verify everything is working:

1. **Load environment:**
   ```bash
   source load_env.sh
   ```

2. **Test alerts:**
   ```bash
   python test_setup_lifecycle.py
   ```

3. **Check dashboard:**
   - Open the dashboard
   - Look for "✅ Telegram alerts are configured and active"
   - Click "🔔 Test Telegram Alert" button

4. **Check tracker:**
   - Run the tracker
   - Create a setup
   - Watch for Telegram alerts

## 🔒 Security Notes

- Keep your `.env` file secure and don't commit it to version control
- The `.env` file is already in `.gitignore`
- Never share your bot token publicly
- Consider using different bots for development and production

## 🎯 Quick Start

1. **Create .env file:**
   ```bash
   cp env_template.txt .env
   # Edit .env with your bot credentials
   ```

2. **Load environment:**
   ```bash
   source load_env.sh
   ```

3. **Run dashboard:**
   ```bash
   streamlit run src/dashboard/app.py
   ```

4. **Test alerts:**
   - Click "🔔 Test Telegram Alert" in the dashboard
   - Or run: `python test_setup_lifecycle.py`

That's it! Your Telegram alerts should now be working! 🎉
