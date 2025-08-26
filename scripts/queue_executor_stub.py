#!/usr/bin/env python3
"""
Queue executor stub for processing Telegram bot commands.

This script watches the command queue directory for JSON job files and processes them.
For now, it just logs the actions and optionally updates setup statuses in the CSV.
"""

import os
import json
import time
import logging
import glob
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Environment variables
RUNS_DIR = os.getenv('RUNS_DIR', 'runs')
SETUPS_CSV = os.getenv('SETUPS_CSV', os.path.join(RUNS_DIR, 'setups.csv'))
CMD_QUEUE_DIR = os.getenv('CMD_QUEUE_DIR', os.path.join(RUNS_DIR, 'command_queue'))

def load_setups_df() -> pd.DataFrame:
    """Load setups CSV."""
    try:
        if not os.path.exists(SETUPS_CSV):
            return pd.DataFrame()
        return pd.read_csv(SETUPS_CSV)
    except Exception as e:
        logger.error(f"Error loading setups: {e}")
        return pd.DataFrame()

def save_setups_df(df: pd.DataFrame):
    """Save setups CSV."""
    try:
        df.to_csv(SETUPS_CSV, index=False)
        logger.info(f"Saved setups CSV with {len(df)} rows")
    except Exception as e:
        logger.error(f"Error saving setups: {e}")

def get_setup_by_id(df: pd.DataFrame, setup_id: str) -> Optional[pd.Series]:
    """Get setup by unique_id or id."""
    try:
        # Try unique_id first
        mask = df['unique_id'] == setup_id
        if mask.any():
            return df[mask].iloc[0]
        
        # Fallback to id
        mask = df['id'] == setup_id
        if mask.any():
            return df[mask].iloc[0]
        
        return None
    except Exception as e:
        logger.error(f"Error getting setup {setup_id}: {e}")
        return None

def process_execute_job(job_data: Dict[str, Any]) -> bool:
    """Process an execute job."""
    setup_id = job_data.get('setup_id')
    requested_by = job_data.get('requested_by', 'unknown')
    
    logger.info(f"🚀 Processing EXECUTE job for setup {setup_id} requested by {requested_by}")
    
    # Load setups
    df = load_setups_df()
    if df.empty:
        logger.error("No setups found")
        return False
    
    # Find setup
    setup = get_setup_by_id(df, setup_id)
    if setup is None or setup.empty:
        logger.error(f"Setup {setup_id} not found")
        return False
    
    # Check if setup should be skipped (superseded or canceled by resolver)
    try:
        from src.utils.setup_resolver import should_skip_execution
        if should_skip_execution(setup_id):
            logger.warning(f"Setup {setup_id} skipped - superseded or canceled by resolver")
            return False
    except Exception as e:
        logger.warning(f"Error checking setup skip status: {e}")
        # Continue with execution if resolver check fails
    
    # Check if still pending
    status = str(setup.get('status', '')).lower()
    if status not in ['pending', 'pending']:
        logger.warning(f"Setup {setup_id} is not pending (status: {status})")
        return False
    
    # Update status to "executed" (or "queued" if you prefer)
    mask = df['unique_id'] == setup_id
    if not mask.any():
        mask = df['id'] == setup_id
    
    if mask.any():
        df.loc[mask, 'status'] = 'executed'
        df.loc[mask, 'executed_at'] = datetime.now().isoformat()
        df.loc[mask, 'executed_by'] = requested_by
        save_setups_df(df)
        
        logger.info(f"✅ Setup {setup_id} marked as executed")
        return True
    else:
        logger.error(f"Could not find setup {setup_id} to update")
        return False

def process_cancel_job(job_data: Dict[str, Any]) -> bool:
    """Process a cancel job."""
    setup_id = job_data.get('setup_id')
    requested_by = job_data.get('requested_by', 'unknown')
    
    logger.info(f"❌ Processing CANCEL job for setup {setup_id} requested by {requested_by}")
    
    # Load setups
    df = load_setups_df()
    if df.empty:
        logger.error("No setups found")
        return False
    
    # Find setup
    setup = get_setup_by_id(df, setup_id)
    if setup is None or setup.empty:
        logger.error(f"Setup {setup_id} not found")
        return False
    
    # Check if setup should be skipped (superseded or canceled by resolver)
    try:
        from src.utils.setup_resolver import should_skip_execution
        if should_skip_execution(setup_id):
            logger.warning(f"Setup {setup_id} skipped - superseded or canceled by resolver")
            return False
    except Exception as e:
        logger.warning(f"Error checking setup skip status: {e}")
        # Continue with cancellation if resolver check fails
    
    # Check if can be cancelled (pending, executed, or triggered)
    status = str(setup.get('status', '')).lower()
    if status not in ['pending', 'pending', 'executed', 'triggered']:
        logger.warning(f"Setup {setup_id} cannot be cancelled (status: {status})")
        return False
    
    # Update status to "cancelled"
    mask = df['unique_id'] == setup_id
    if not mask.any():
        mask = df['id'] == setup_id
    
    if mask.any():
        df.loc[mask, 'status'] = 'cancelled'
        df.loc[mask, 'cancelled_at'] = datetime.now().isoformat()
        df.loc[mask, 'cancelled_by'] = requested_by
        save_setups_df(df)
        
        logger.info(f"✅ Setup {setup_id} marked as cancelled")
        return True
    else:
        logger.error(f"Could not find setup {setup_id} to update")
        return False

def process_job_file(job_file: str) -> bool:
    """Process a single job file."""
    try:
        # Read job data
        with open(job_file, 'r') as f:
            job_data = json.load(f)
        
        # Check expiry
        expires_at = job_data.get('expires_at')
        if expires_at:
            try:
                expiry_ts = datetime.fromisoformat(expires_at)
                if datetime.now() > expiry_ts:
                    logger.warning(f"Job {job_file} has expired")
                    os.remove(job_file)
                    return False
            except Exception as e:
                logger.warning(f"Error parsing expiry for {job_file}: {e}")
        
        # Process based on action
        action = job_data.get('action')
        if action in ['execute', 'exec']:  # Handle both full and shortened action names
            success = process_execute_job(job_data)
        elif action == 'cancel':
            success = process_cancel_job(job_data)
        else:
            logger.error(f"Unknown action: {action}")
            success = False
        
        # Remove processed job file
        if success:
            os.remove(job_file)
            logger.info(f"✅ Processed and removed job file: {job_file}")
        else:
            # Move to error folder for investigation
            error_dir = os.path.join(CMD_QUEUE_DIR, 'errors')
            os.makedirs(error_dir, exist_ok=True)
            error_file = os.path.join(error_dir, os.path.basename(job_file))
            os.rename(job_file, error_file)
            logger.error(f"❌ Moved failed job to: {error_file}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error processing job file {job_file}: {e}")
        return False

def cleanup_expired_idempotency_files():
    """Clean up expired idempotency marker files."""
    try:
        idemp_pattern = os.path.join(CMD_QUEUE_DIR, 'idemp_*.json')
        for idemp_file in glob.glob(idemp_pattern):
            try:
                # Check if file is older than 1 hour
                if time.time() - os.path.getmtime(idemp_file) > 3600:
                    os.remove(idemp_file)
                    logger.debug(f"Cleaned up expired idempotency file: {idemp_file}")
            except Exception as e:
                logger.warning(f"Error cleaning up {idemp_file}: {e}")
    except Exception as e:
        logger.warning(f"Error during idempotency cleanup: {e}")

def main():
    """Main loop."""
    logger.info("Starting queue executor stub...")
    logger.info(f"Watching directory: {CMD_QUEUE_DIR}")
    logger.info(f"Setups CSV: {SETUPS_CSV}")
    
    # Ensure directories exist
    os.makedirs(CMD_QUEUE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SETUPS_CSV), exist_ok=True)
    
    last_cleanup = time.time()
    
    while True:
        try:
            # Clean up expired idempotency files every hour
            if time.time() - last_cleanup > 3600:
                cleanup_expired_idempotency_files()
                last_cleanup = time.time()
            
            # Look for job files
            job_pattern = os.path.join(CMD_QUEUE_DIR, '*.json')
            job_files = glob.glob(job_pattern)
            
            # Filter out idempotency files
            job_files = [f for f in job_files if not os.path.basename(f).startswith('idemp_')]
            
            if job_files:
                logger.info(f"Found {len(job_files)} job file(s) to process")
                
                for job_file in job_files:
                    process_job_file(job_file)
            else:
                # Sleep longer when no jobs
                time.sleep(5)
                continue
            
            # Short sleep between processing
            time.sleep(1)
            
        except KeyboardInterrupt:
            logger.info("Shutting down queue executor...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
