#!/usr/bin/env python3
"""
Trading logger for alpha12_24
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import os


@dataclass
class TradeLog:
    """Trade log entry"""
    timestamp: datetime
    signal: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    leverage: float
    confidence: float
    risk_reward: float
    asset: str
    status: str  # 'open', 'closed', 'cancelled'
    metadata: Dict


@dataclass
class SignalLog:
    """Signal log entry"""
    timestamp: datetime
    signal: str
    confidence: float
    prob_up: float
    prob_down: float
    asset: str
    regime: str
    volatility: float
    metadata: Dict


class TradingLogger:
    """Trading logger for alpha12_24"""
    
    def __init__(self, config, log_dir: str = "logs"):
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize logging
        self.setup_logging()
        
        # Initialize data storage
        self.trades = []
        self.signals = []
        self.performance_metrics = {}
        
        # Create log files
        self.trades_file = self.log_dir / "trades.jsonl"
        self.signals_file = self.log_dir / "signals.jsonl"
        self.performance_file = self.log_dir / "performance.json"
    
    def setup_logging(self):
        """Setup logging configuration"""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handler
        log_file = self.log_dir / "trading.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger('alpha12_24')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_signal(self, signal: str, confidence: float, prob_up: float, 
                  prob_down: float, asset: str, regime: str, volatility: float,
                  metadata: Dict) -> None:
        """
        Log trading signal
        
        Args:
            signal: Trading signal
            confidence: Signal confidence
            prob_up: Probability of up movement
            prob_down: Probability of down movement
            asset: Trading asset
            regime: Market regime
            volatility: Current volatility
            metadata: Additional metadata
        """
        signal_log = SignalLog(
            timestamp=datetime.now(),
            signal=signal,
            confidence=confidence,
            prob_up=prob_up,
            prob_down=prob_down,
            asset=asset,
            regime=regime,
            volatility=volatility,
            metadata=metadata
        )
        
        # Add to memory
        self.signals.append(signal_log)
        
        # Write to file
        self._write_signal_to_file(signal_log)
        
        # Log to console
        self.logger.info(
            f"Signal: {signal} | Confidence: {confidence:.2f} | "
            f"Asset: {asset} | Regime: {regime}"
        )
    
    def log_trade(self, signal: str, entry_price: float, stop_loss: float,
                 take_profit: float, position_size: float, leverage: float,
                 confidence: float, risk_reward: float, asset: str,
                 metadata: Dict) -> str:
        """
        Log new trade
        
        Args:
            signal: Trading signal
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            position_size: Position size
            leverage: Leverage used
            confidence: Signal confidence
            risk_reward: Risk/reward ratio
            asset: Trading asset
            metadata: Additional metadata
        
        Returns:
            Trade ID
        """
        trade_id = f"trade_{len(self.trades) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        trade_log = TradeLog(
            timestamp=datetime.now(),
            signal=signal,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            leverage=leverage,
            confidence=confidence,
            risk_reward=risk_reward,
            asset=asset,
            status='open',
            metadata={**metadata, 'trade_id': trade_id}
        )
        
        # Add to memory
        self.trades.append(trade_log)
        
        # Write to file
        self._write_trade_to_file(trade_log)
        
        # Log to console
        self.logger.info(
            f"New Trade: {trade_id} | {signal} {asset} | "
            f"Entry: {entry_price:.2f} | Size: {position_size:.4f} | "
            f"RR: {risk_reward:.2f}"
        )
        
        return trade_id
    
    def update_trade_status(self, trade_id: str, status: str, 
                          exit_price: Optional[float] = None,
                          exit_reason: Optional[str] = None) -> None:
        """
        Update trade status
        
        Args:
            trade_id: Trade ID
            status: New status ('closed', 'cancelled')
            exit_price: Exit price (for closed trades)
            exit_reason: Reason for exit
        """
        # Find trade in memory
        for trade in self.trades:
            if trade.metadata.get('trade_id') == trade_id:
                trade.status = status
                if exit_price is not None:
                    trade.metadata['exit_price'] = exit_price
                if exit_reason is not None:
                    trade.metadata['exit_reason'] = exit_reason
                trade.metadata['exit_timestamp'] = datetime.now()
                
                # Log to console
                if status == 'closed':
                    pnl = self._calculate_trade_pnl(trade)
                    self.logger.info(
                        f"Trade Closed: {trade_id} | P&L: {pnl:.2f} | "
                        f"Exit Price: {exit_price:.2f} | Reason: {exit_reason}"
                    )
                else:
                    self.logger.info(f"Trade Cancelled: {trade_id}")
                
                break
    
    def _calculate_trade_pnl(self, trade: TradeLog) -> float:
        """
        Calculate trade P&L
        
        Args:
            trade: Trade log entry
        
        Returns:
            P&L value
        """
        if trade.status != 'closed' or 'exit_price' not in trade.metadata:
            return 0.0
        
        exit_price = trade.metadata['exit_price']
        
        if trade.signal == "LONG":
            pnl = (exit_price - trade.entry_price) * trade.position_size
        else:  # SHORT
            pnl = (trade.entry_price - exit_price) * trade.position_size
        
        return pnl
    
    def _write_signal_to_file(self, signal_log: SignalLog) -> None:
        """
        Write signal to file
        
        Args:
            signal_log: Signal log entry
        """
        try:
            with open(self.signals_file, 'a') as f:
                signal_dict = asdict(signal_log)
                signal_dict['timestamp'] = signal_dict['timestamp'].isoformat()
                f.write(json.dumps(signal_dict) + '\n')
        except Exception as e:
            self.logger.error(f"Error writing signal to file: {e}")
    
    def _write_trade_to_file(self, trade_log: TradeLog) -> None:
        """
        Write trade to file
        
        Args:
            trade_log: Trade log entry
        """
        try:
            with open(self.trades_file, 'a') as f:
                trade_dict = asdict(trade_log)
                trade_dict['timestamp'] = trade_dict['timestamp'].isoformat()
                f.write(json.dumps(trade_dict) + '\n')
        except Exception as e:
            self.logger.error(f"Error writing trade to file: {e}")
    
    def load_trades_from_file(self) -> List[TradeLog]:
        """
        Load trades from file
        
        Returns:
            List of trade logs
        """
        trades = []
        
        if not self.trades_file.exists():
            return trades
        
        try:
            with open(self.trades_file, 'r') as f:
                for line in f:
                    trade_dict = json.loads(line.strip())
                    trade_dict['timestamp'] = datetime.fromisoformat(trade_dict['timestamp'])
                    trades.append(TradeLog(**trade_dict))
        except Exception as e:
            self.logger.error(f"Error loading trades from file: {e}")
        
        return trades
    
    def load_signals_from_file(self) -> List[SignalLog]:
        """
        Load signals from file
        
        Returns:
            List of signal logs
        """
        signals = []
        
        if not self.signals_file.exists():
            return signals
        
        try:
            with open(self.signals_file, 'r') as f:
                for line in f:
                    signal_dict = json.loads(line.strip())
                    signal_dict['timestamp'] = datetime.fromisoformat(signal_dict['timestamp'])
                    signals.append(SignalLog(**signal_dict))
        except Exception as e:
            self.logger.error(f"Error loading signals from file: {e}")
        
        return signals
    
    def get_trade_summary(self) -> Dict:
        """
        Get trade summary statistics
        
        Returns:
            Dictionary with trade summary
        """
        if not self.trades:
            return {}
        
        closed_trades = [t for t in self.trades if t.status == 'closed']
        open_trades = [t for t in self.trades if t.status == 'open']
        
        summary = {
            'total_trades': len(self.trades),
            'closed_trades': len(closed_trades),
            'open_trades': len(open_trades),
            'long_trades': len([t for t in self.trades if t.signal == 'LONG']),
            'short_trades': len([t for t in self.trades if t.signal == 'SHORT'])
        }
        
        if closed_trades:
            pnls = [self._calculate_trade_pnl(t) for t in closed_trades]
            summary.update({
                'total_pnl': sum(pnls),
                'avg_pnl': np.mean(pnls),
                'win_rate': len([p for p in pnls if p > 0]) / len(pnls),
                'max_profit': max(pnls),
                'max_loss': min(pnls),
                'pnl_std': np.std(pnls)
            })
        
        return summary
    
    def get_signal_summary(self) -> Dict:
        """
        Get signal summary statistics
        
        Returns:
            Dictionary with signal summary
        """
        if not self.signals:
            return {}
        
        recent_signals = self.signals[-100:]  # Last 100 signals
        
        summary = {
            'total_signals': len(self.signals),
            'recent_signals': len(recent_signals),
            'long_signals': len([s for s in self.signals if s.signal == 'LONG']),
            'short_signals': len([s for s in self.signals if s.signal == 'SHORT']),
            'hold_signals': len([s for s in self.signals if s.signal == 'HOLD']),
            'avg_confidence': np.mean([s.confidence for s in self.signals]),
            'avg_volatility': np.mean([s.volatility for s in self.signals])
        }
        
        return summary
    
    def export_trades_to_csv(self, filename: Optional[str] = None) -> str:
        """
        Export trades to CSV
        
        Args:
            filename: Output filename (optional)
        
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"trades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = self.log_dir / filename
        
        try:
            # Convert trades to DataFrame
            trades_data = []
            for trade in self.trades:
                trade_dict = asdict(trade)
                trade_dict['timestamp'] = trade_dict['timestamp'].isoformat()
                trades_data.append(trade_dict)
            
            df = pd.DataFrame(trades_data)
            df.to_csv(filepath, index=False)
            
            self.logger.info(f"Trades exported to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error exporting trades to CSV: {e}")
            return ""
    
    def export_signals_to_csv(self, filename: Optional[str] = None) -> str:
        """
        Export signals to CSV
        
        Args:
            filename: Output filename (optional)
        
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"signals_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = self.log_dir / filename
        
        try:
            # Convert signals to DataFrame
            signals_data = []
            for signal in self.signals:
                signal_dict = asdict(signal)
                signal_dict['timestamp'] = signal_dict['timestamp'].isoformat()
                signals_data.append(signal_dict)
            
            df = pd.DataFrame(signals_data)
            df.to_csv(filepath, index=False)
            
            self.logger.info(f"Signals exported to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error exporting signals to CSV: {e}")
            return ""
