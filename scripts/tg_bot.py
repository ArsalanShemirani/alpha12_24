#!/usr/bin/env python3
"""
Minimal Telegram bot for setup execution/cancel with reply-to support.

Commands:
- /pending: List pending setups with inline buttons
- /execute <setup_id>: Execute a setup (with reply-to support)
- /cancel <setup_id>: Cancel a setup (with reply-to support)

Features:
- Allowlist-based access control
- Reply-to message parsing for setup_id extraction
- 2-step confirmation with inline keyboard
- Idempotency with expiry
- Audit logging
- Filesystem-based command queue
"""

import os
import re
import json
import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.constants import ParseMode

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Environment variables
TG_BOT_TOKEN = os.getenv('TG_BOT_TOKEN')
TG_CHAT_ID = os.getenv('TG_CHAT_ID')
TG_ALLOWLIST = os.getenv('TG_ALLOWLIST', '')

# File paths
RUNS_DIR = os.getenv('RUNS_DIR', 'runs')
SETUPS_CSV = os.getenv('SETUPS_CSV', os.path.join(RUNS_DIR, 'setups.csv'))
CMD_QUEUE_DIR = os.getenv('CMD_QUEUE_DIR', os.path.join(RUNS_DIR, 'command_queue'))
TELEGRAM_AUDIT = os.getenv('TELEGRAM_AUDIT', os.path.join(RUNS_DIR, 'telegram_audit.csv'))

# Ensure directories exist
os.makedirs(CMD_QUEUE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(TELEGRAM_AUDIT), exist_ok=True)

# Parse allowlist
ALLOWED_CHAT_IDS = set()
if TG_ALLOWLIST:
    ALLOWED_CHAT_IDS = {int(x.strip()) for x in TG_ALLOWLIST.split(',') if x.strip()}

# Add main chat ID if not in allowlist
if TG_CHAT_ID:
    ALLOWED_CHAT_IDS.add(int(TG_CHAT_ID))

# Debug logging for allowlist
logger.info(f"TG_CHAT_ID: {TG_CHAT_ID}")
logger.info(f"TG_ALLOWLIST: {TG_ALLOWLIST}")
logger.info(f"ALLOWED_CHAT_IDS: {ALLOWED_CHAT_IDS}")

def is_allowed_user(chat_id: int) -> bool:
    """Check if user is in allowlist."""
    return chat_id in ALLOWED_CHAT_IDS

def extract_setup_id_from_text(text: str) -> Optional[str]:
    """Extract setup ID from text using regex."""
    if not text:
        return None
    
    # Look for "Setup ID: <id>" pattern
    match = re.search(r'^Setup ID:\s*(\S+)', text, re.MULTILINE)
    if match:
        return match.group(1)
    
    # Also try to match the unique_id pattern directly
    match = re.search(r'(AUTO|MANUAL)-[A-Z]+-\d+[hmd]-[A-Z]+-\d{8}-\d+', text)
    if match:
        return match.group(0)  # Return the full match, not just group 1
    
    return None

def load_pending_setups() -> pd.DataFrame:
    """Load pending setups from CSV."""
    try:
        if not os.path.exists(SETUPS_CSV):
            return pd.DataFrame()
        
        df = pd.read_csv(SETUPS_CSV)
        if df.empty:
            return df
        
        # Filter for pending setups (case-insensitive)
        pending_mask = df['status'].str.lower().isin(['pending', 'pending'])
        return df[pending_mask].sort_values('created_at', ascending=False).head(10)
    
    except Exception as e:
        logger.error(f"Error loading setups: {e}")
        return pd.DataFrame()

def load_active_setups() -> pd.DataFrame:
    """Load active setups (executed, triggered) from CSV."""
    try:
        if not os.path.exists(SETUPS_CSV):
            return pd.DataFrame()
        
        df = pd.read_csv(SETUPS_CSV)
        if df.empty:
            return df
        
        # Filter for active setups (executed, triggered)
        active_mask = df['status'].str.lower().isin(['executed', 'triggered'])
        return df[active_mask].sort_values('created_at', ascending=False).head(10)
    
    except Exception as e:
        logger.error(f"Error loading active setups: {e}")
        return pd.DataFrame()

def get_setup_by_id(setup_id: str) -> Optional[pd.Series]:
    """Get setup by unique_id."""
    try:
        df = pd.read_csv(SETUPS_CSV)
        if df.empty:
            return None
        
        # Try to match by unique_id first
        mask = df['unique_id'] == setup_id
        if mask.any():
            return df[mask].iloc[0]
        
        # Fallback to id column
        mask = df['id'] == setup_id
        if mask.any():
            return df[mask].iloc[0]
        
        return None
    
    except Exception as e:
        logger.error(f"Error getting setup {setup_id}: {e}")
        return None

def format_setup_card(setup: pd.Series) -> str:
    """Format setup as a card for display."""
    try:
        asset = setup.get('asset', 'N/A')
        interval = setup.get('interval', 'N/A')
        direction = setup.get('direction', 'N/A').upper()
        unique_id = setup.get('unique_id', 'N/A')
        entry = setup.get('entry', 0)
        stop = setup.get('stop', 0)
        target = setup.get('target', 0)
        rr = setup.get('rr', 0)
        confidence = setup.get('confidence', 0)
        valid_until = setup.get('valid_until', 'N/A')
        
        # Format numbers
        entry_str = f"${entry:.2f}" if entry else "N/A"
        stop_str = f"${stop:.2f}" if stop else "N/A"
        target_str = f"${target:.2f}" if target else "N/A"
        rr_str = f"{rr:.2f}" if rr else "N/A"
        confidence_str = f"{confidence:.1f}%" if confidence else "N/A"
        
        # Format valid_until
        if pd.notna(valid_until) and valid_until != 'N/A':
            try:
                valid_ts = pd.to_datetime(valid_until)
                valid_str = valid_ts.strftime('%Y-%m-%d %H:%M')
            except:
                valid_str = str(valid_until)
        else:
            valid_str = "N/A"
        
        card = f"""<b>{asset} {interval} ({direction})</b>
Setup ID: <code>{unique_id}</code>
Entry: {entry_str} | SL: {stop_str} | TP: {target_str}
RR: {rr_str} | Conf: {confidence_str}
Valid until: {valid_str}"""
        
        return card
    
    except Exception as e:
        logger.error(f"Error formatting setup card: {e}")
        return f"Error formatting setup: {e}"

def create_confirmation_keyboard(action: str, setup_id: str, expiry_minutes: int = 1) -> InlineKeyboardMarkup:
    """Create confirmation keyboard with expiry."""
    idempotency_key = str(uuid.uuid4())[:8]  # Shorter key
    
    # Create shorter callback data: action|setup_id|idemp
    callback_data = f"{action}|{setup_id}|{idempotency_key}"
    
    keyboard = [
        [
            InlineKeyboardButton("✅ Confirm", callback_data=callback_data),
            InlineKeyboardButton("❌ Abort", callback_data="abort")
        ]
    ]
    
    return InlineKeyboardMarkup(keyboard)

def write_command_job(action: str, setup_id: str, requested_by: str, idempotency_key: str, 
                     expires_at: datetime) -> str:
    """Write command job to filesystem queue."""
    job_data = {
        "action": action,
        "setup_id": setup_id,
        "requested_by": requested_by,
        "idempotency_key": idempotency_key,
        "expires_at": expires_at.isoformat(),
        "created_at": datetime.now().isoformat()
    }
    
    # Create filename: {ts}_{action}_{setup_id}_{nonce}.json
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    nonce = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{action}_{setup_id.replace('-', '_')}_{nonce}.json"
    filepath = os.path.join(CMD_QUEUE_DIR, filename)
    
    # Write job file
    with open(filepath, 'w') as f:
        json.dump(job_data, f, indent=2)
    
    return filename

def append_audit_log(user_id: int, username: str, action: str, setup_id: str, 
                    idempotency_key: str, result: str):
    """Append audit log to CSV."""
    try:
        audit_data = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'username': username or 'unknown',
            'action': action,
            'setup_id': setup_id,
            'idempotency_key': idempotency_key,
            'result': result
        }
        
        # Create audit file if it doesn't exist
        if not os.path.exists(TELEGRAM_AUDIT):
            with open(TELEGRAM_AUDIT, 'w') as f:
                f.write('timestamp,user_id,username,action,setup_id,idempotency_key,result\n')
        
        # Append audit entry
        with open(TELEGRAM_AUDIT, 'a') as f:
            f.write(f'{audit_data["timestamp"]},{audit_data["user_id"]},{audit_data["username"]},'
                   f'{audit_data["action"]},{audit_data["setup_id"]},{audit_data["idempotency_key"]},'
                   f'{audit_data["result"]}\n')
    
    except Exception as e:
        logger.error(f"Error writing audit log: {e}")

async def pending_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /pending command."""
    if not is_allowed_user(update.effective_chat.id):
        await update.message.reply_text("❌ Access denied. You are not authorized to use this bot.")
        return
    
    setups = load_pending_setups()
    
    if setups.empty:
        await update.message.reply_text("📭 No pending setups found.")
        return
    
    # Send each setup as a separate message with inline buttons
    for _, setup in setups.iterrows():
        card_text = format_setup_card(setup)
        setup_id = setup.get('unique_id', setup.get('id', 'N/A'))
        
        # Create shorter callback data to avoid Telegram's 64-byte limit
        exec_id = str(uuid.uuid4())[:8]
        cancel_id = str(uuid.uuid4())[:8]
        
        keyboard = [
            [
                InlineKeyboardButton("🚀 Execute", callback_data=f"exec|{setup_id}|{exec_id}"),
                InlineKeyboardButton("❌ Cancel", callback_data=f"cancel|{setup_id}|{cancel_id}")
            ]
        ]
        
        await update.message.reply_text(
            card_text,
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    await update.message.reply_text(f"📋 Found {len(setups)} pending setup(s)")

async def active_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /active command."""
    if not is_allowed_user(update.effective_chat.id):
        await update.message.reply_text("❌ Access denied. You are not authorized to use this bot.")
        return
    
    setups = load_active_setups()
    
    if setups.empty:
        await update.message.reply_text("📭 No active setups found.")
        return
    
    # Send each setup as a separate message with inline buttons
    for _, setup in setups.iterrows():
        card_text = format_setup_card(setup)
        setup_id = setup.get('unique_id', setup.get('id', 'N/A'))
        status = setup.get('status', 'N/A').upper()
        
        # Create shorter callback data to avoid Telegram's 64-byte limit
        cancel_id = str(uuid.uuid4())[:8]
        
        keyboard = [
            [
                InlineKeyboardButton("❌ Cancel", callback_data=f"cancel|{setup_id}|{cancel_id}")
            ]
        ]
        
        # Add status to the card
        status_card = f"{card_text}\n\n<b>Status: {status}</b>"
        
        await update.message.reply_text(
            status_card,
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    await update.message.reply_text(f"📋 Found {len(setups)} active setup(s)")

async def execute_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /execute command."""
    if not is_allowed_user(update.effective_chat.id):
        await update.message.reply_text("❌ Access denied. You are not authorized to use this bot.")
        return
    
    # Extract setup_id from command or reply
    setup_id = None
    if context.args:
        setup_id = context.args[0]
    elif update.message.reply_to_message:
        setup_id = extract_setup_id_from_text(update.message.reply_to_message.text)
    
    if not setup_id:
        await update.message.reply_text(
            "❌ Please provide a setup ID or reply to a setup message.\n"
            "Usage: /execute <setup_id> or reply to a setup with /execute"
        )
        return
    
    # Get setup details
    setup = get_setup_by_id(setup_id)
    if not setup:
        await update.message.reply_text(f"❌ Setup not found: {setup_id}")
        return
    
    # Check if setup is pending
    status = setup.get('status', '').lower()
    if status not in ['pending', 'pending']:
        await update.message.reply_text(f"❌ Setup {setup_id} is not pending (status: {status})")
        return
    
    # Show confirmation
    card_text = format_setup_card(setup)
    confirmation_text = f"🚀 <b>Confirm Execution</b>\n\n{card_text}\n\nAre you sure you want to execute this setup?"
    
    keyboard = create_confirmation_keyboard("execute", setup_id)
    
    await update.message.reply_text(
        confirmation_text,
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard
    )

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /cancel command."""
    if not is_allowed_user(update.effective_chat.id):
        await update.message.reply_text("❌ Access denied. You are not authorized to use this bot.")
        return
    
    # Extract setup_id from command or reply
    setup_id = None
    if context.args:
        setup_id = context.args[0]
    elif update.message.reply_to_message:
        setup_id = extract_setup_id_from_text(update.message.reply_to_message.text)
    
    if not setup_id:
        await update.message.reply_text(
            "❌ Please provide a setup ID or reply to a setup message.\n"
            "Usage: /cancel <setup_id> or reply to a setup with /cancel"
        )
        return
    
    # Get setup details
    setup = get_setup_by_id(setup_id)
    if not setup:
        await update.message.reply_text(f"❌ Setup not found: {setup_id}")
        return
    
    # Check if setup can be cancelled (pending, executed, or triggered)
    status = setup.get('status', '').lower()
    if status not in ['pending', 'pending', 'executed', 'triggered']:
        await update.message.reply_text(f"❌ Setup {setup_id} cannot be cancelled (status: {status})")
        return
    
    # Show confirmation
    card_text = format_setup_card(setup)
    confirmation_text = f"❌ <b>Confirm Cancellation</b>\n\n{card_text}\n\nAre you sure you want to cancel this setup?"
    
    keyboard = create_confirmation_keyboard("cancel", setup_id)
    
    await update.message.reply_text(
        confirmation_text,
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline button callbacks."""
    query = update.callback_query
    await query.answer()
    
    # Debug logging
    logger.info(f"Button callback from chat_id: {query.message.chat.id}, user_id: {query.from_user.id}")
    logger.info(f"Allowed chat IDs: {ALLOWED_CHAT_IDS}")
    
    # Check both chat_id and user_id for authorization
    if not is_allowed_user(query.message.chat.id) and not is_allowed_user(query.from_user.id):
        await query.edit_message_text("❌ Access denied. You are not authorized to use this bot.")
        return
    
    if query.data == "abort":
        await query.edit_message_text("❌ Operation aborted.")
        return
    
    # Parse callback data: action|setup_id|idemp
    try:
        parts = query.data.split('|')
        if len(parts) != 3:
            await query.edit_message_text("❌ Invalid callback data.")
            return
        
        action, setup_id, idempotency_key = parts
        
        # Check idempotency (simple file check)
        idempotency_file = os.path.join(CMD_QUEUE_DIR, f"idemp_{idempotency_key}.json")
        if os.path.exists(idempotency_file):
            await query.edit_message_text("✅ Command already processed (idempotency check).")
            return
        
        # Get setup details
        setup = get_setup_by_id(setup_id)
        if setup is None or setup.empty:
            await query.edit_message_text(f"❌ Setup not found: {setup_id}")
            return
        
        # Check if setup can be processed
        status = str(setup.get('status', '')).lower()
        if action == "exec" and status not in ['pending', 'pending']:
            await query.edit_message_text(f"❌ Setup {setup_id} is not pending (status: {status})")
            return
        elif action == "cancel" and status not in ['pending', 'pending', 'executed', 'triggered']:
            await query.edit_message_text(f"❌ Setup {setup_id} cannot be cancelled (status: {status})")
            return
        
        # Write command job
        expires_at = datetime.now() + timedelta(minutes=5)
        job_filename = write_command_job(
            action, setup_id, 
            f"{query.from_user.username or query.from_user.id}",
            idempotency_key, expires_at
        )
        
        # Create idempotency marker
        with open(idempotency_file, 'w') as f:
            json.dump({"processed_at": datetime.now().isoformat()}, f)
        
        # Log audit
        append_audit_log(
            query.from_user.id,
            query.from_user.username,
            action,
            setup_id,
            idempotency_key,
            "queued"
        )
        
        # Update message
        action_emoji = "🚀" if action == "execute" else "❌"
        await query.edit_message_text(
            f"{action_emoji} ✅ Command queued successfully!\n"
            f"Setup: {setup_id}\n"
            f"Action: {action}\n"
            f"Job: {job_filename}"
        )
        
    except Exception as e:
        logger.error(f"Error processing callback: {e}")
        await query.edit_message_text("❌ Error processing command. Please try again.")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    if not is_allowed_user(update.effective_chat.id):
        await update.message.reply_text("❌ Access denied. You are not authorized to use this bot.")
        return
    
    help_text = """🤖 <b>Alpha12 Trading Bot</b>

Available commands:
• <code>/pending</code> - List pending setups
• <code>/active</code> - List active setups (executed/triggered)
• <code>/execute &lt;setup_id&gt;</code> - Execute a setup
• <code>/cancel &lt;setup_id&gt;</code> - Cancel a setup

You can also reply to setup messages with /execute or /cancel (no arguments needed).

All commands require confirmation via inline buttons."""
    
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)

def main():
    """Main function."""
    if not TG_BOT_TOKEN:
        logger.error("TG_BOT_TOKEN not set")
        return
    
    if not ALLOWED_CHAT_IDS:
        logger.warning("No allowed chat IDs configured")
    
    # Create application
    application = Application.builder().token(TG_BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("pending", pending_command))
    application.add_handler(CommandHandler("active", active_command))
    application.add_handler(CommandHandler("execute", execute_command))
    application.add_handler(CommandHandler("cancel", cancel_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Start bot
    logger.info("Starting Telegram bot...")
    application.run_polling()

if __name__ == "__main__":
    main()
