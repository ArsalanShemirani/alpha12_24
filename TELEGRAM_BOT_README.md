# 🤖 Alpha12 Telegram Bot

A minimal Telegram bot for setup execution/cancel with reply-to support, designed to work with the Alpha12 trading system.

## 📋 Features

- **Allowlist-based access control** - Only authorized users can use the bot
- **Reply-to support** - Execute/cancel setups by replying to setup messages
- **2-step confirmation** - Inline keyboard confirmation with expiry
- **Idempotency** - Prevents duplicate command processing
- **Audit logging** - All commands and results are logged
- **Filesystem-based queue** - Simple, reliable command processing

## 🚀 Commands

### `/pending`
Lists all pending setups (auto + manual) with compact cards showing:
- Asset, timeframe, and direction
- Setup ID
- Entry, Stop Loss, and Take Profit levels
- Risk:Reward ratio and confidence
- Valid until timestamp
- Inline buttons for Execute/Cancel

### `/execute <setup_id>`
Executes a pending setup. You can:
- Provide the setup ID as an argument: `/execute MANUAL-ETHUSDT-4h-LONG-20250825-140000`
- Reply to a setup message with just `/execute` (no arguments needed)

### `/cancel <setup_id>`
Cancels a pending setup. Same usage as `/execute`.

## 🔧 Installation

### 1. Environment Variables

Add these to your `.env` file:

```bash
# Telegram Bot Configuration
TG_BOT_TOKEN=your_bot_token_here
TG_CHAT_ID=your_chat_id_here
TG_ALLOWLIST=chat_id1,chat_id2,chat_id3

# File paths (optional - defaults provided)
RUNS_DIR=runs
SETUPS_CSV=runs/setups.csv
CMD_QUEUE_DIR=runs/command_queue
TELEGRAM_AUDIT=runs/telegram_audit.csv
```

### 2. Install Dependencies

```bash
source venv/bin/activate
pip install python-telegram-bot
```

### 3. Create Directories

```bash
mkdir -p runs/command_queue
```

## 🏃‍♂️ Running the Bot

### Manual Start

```bash
# Start the Telegram bot
source venv/bin/activate
python3 scripts/tg_bot.py

# In another terminal, start the queue executor
source venv/bin/activate
python3 scripts/queue_executor_stub.py
```

### Systemd Services

```bash
# Copy service files
sudo cp alpha12-telegram-bot.service /etc/systemd/system/
sudo cp alpha12-queue-executor.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start services
sudo systemctl enable alpha12-telegram-bot
sudo systemctl enable alpha12-queue-executor
sudo systemctl start alpha12-telegram-bot
sudo systemctl start alpha12-queue-executor

# Check status
sudo systemctl status alpha12-telegram-bot
sudo systemctl status alpha12-queue-executor
```

## 📁 File Structure

```
scripts/
├── tg_bot.py              # Main Telegram bot
└── queue_executor_stub.py # Command queue processor

runs/
├── setups.csv             # Setup data
├── command_queue/         # Job files (auto-created)
│   ├── *.json            # Command jobs
│   └── errors/           # Failed jobs
└── telegram_audit.csv    # Audit log

alpha12-telegram-bot.service      # Systemd service
alpha12-queue-executor.service    # Systemd service
```

## 🔐 Security Features

### Allowlist Control
Only users with chat IDs in the `TG_ALLOWLIST` environment variable can use the bot.

### Confirmation Required
All execute/cancel actions require confirmation via inline keyboard with 60-second expiry.

### Idempotency
Each command has a unique idempotency key to prevent duplicate processing.

### Audit Logging
All commands are logged to `runs/telegram_audit.csv` with:
- Timestamp
- User ID and username
- Action performed
- Setup ID
- Idempotency key
- Result

## 📊 Command Queue Format

Job files are stored as JSON in `runs/command_queue/`:

```json
{
  "action": "execute",
  "setup_id": "MANUAL-ETHUSDT-4h-LONG-20250825-140000",
  "requested_by": "username",
  "idempotency_key": "uuid-here",
  "expires_at": "2025-08-25T14:05:00",
  "created_at": "2025-08-25T14:00:00"
}
```

## 🔄 Queue Executor

The queue executor:
- Watches `runs/command_queue/` for new job files
- Processes execute/cancel commands
- Updates setup statuses in `setups.csv`
- Removes processed job files
- Moves failed jobs to `errors/` folder
- Cleans up expired idempotency files

## 🧪 Testing

Test the bot functionality:

```bash
# Test setup loading and job creation
python3 test_telegram_bot.py

# Check bot logs
sudo journalctl -u alpha12-telegram-bot -f

# Check executor logs
sudo journalctl -u alpha12-queue-executor -f
```

## 📝 Usage Examples

### List Pending Setups
```
/pending
```

### Execute Setup (with ID)
```
/execute MANUAL-ETHUSDT-4h-LONG-20250825-140000
```

### Execute Setup (reply to message)
```
[Reply to setup message with] /execute
```

### Cancel Setup
```
/cancel AUTO-BTCUSDT-4h-SHORT-20250825-120000
```

## ⚠️ Important Notes

1. **No Exchange Integration**: This is a stub implementation. The queue executor only updates CSV statuses, it doesn't actually place trades.

2. **Setup Status Flow**: 
   - Execute: `pending` → `executed`
   - Cancel: `pending` → `cancelled`

3. **File Permissions**: Ensure the `ubuntu` user has write access to `runs/` directory.

4. **Environment Variables**: All paths default to reasonable values if not set in `.env`.

5. **Error Handling**: Failed jobs are moved to `runs/command_queue/errors/` for investigation.

## 🔧 Troubleshooting

### Bot Not Responding
```bash
# Check if bot is running
sudo systemctl status alpha12-telegram-bot

# Check logs
sudo journalctl -u alpha12-telegram-bot -f

# Verify environment variables
grep TG_ .env
```

### Commands Not Processing
```bash
# Check queue executor
sudo systemctl status alpha12-queue-executor

# Check queue directory
ls -la runs/command_queue/

# Check audit log
tail -f runs/telegram_audit.csv
```

### Permission Issues
```bash
# Fix permissions
sudo chown -R ubuntu:ubuntu runs/
chmod 755 runs/command_queue/
```

## 🚀 Next Steps

1. **Exchange Integration**: Replace the queue executor stub with real KuCoin API integration
2. **Redis Queue**: Replace filesystem queue with Redis for better performance
3. **Webhook Support**: Add webhook support for better reliability
4. **Advanced Features**: Add position monitoring, PnL tracking, etc.
