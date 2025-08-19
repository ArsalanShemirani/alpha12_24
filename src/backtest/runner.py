#!/usr/bin/env python3
"""
Backtest runner for alpha12_24
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ..core.config import config
from ..features.engine import FeatureEngine
from ..features.macro import MacroFeatures
from ..models.train import ModelTrainer, TrainedModel
from ..policy.thresholds import ThresholdManager
from ..policy.regime import RegimeDetector
from ..trading.planner import TradingPlanner
from ..trading.leverage import LeverageManager
from ..trading.logger import TradingLogger


@dataclass
class BacktestResult:
    """Backtest result structure"""
    trades: List[Dict]
    signals: List[Dict]
    performance_metrics: Dict
    model_metrics: Dict
    regime_analysis: Dict
    timestamp: datetime


class BacktestRunner:
    """Backtest runner for alpha12_24"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize components
        self.feature_engine = FeatureEngine()
        self.macro_features = MacroFeatures()
        self.model_trainer = ModelTrainer(config)
        self.threshold_manager = ThresholdManager(config)
        self.regime_detector = RegimeDetector(config)
        self.trading_planner = TradingPlanner(config)
        self.leverage_manager = LeverageManager(config)
        self.trading_logger = TradingLogger(config)
        
        # Backtest state
        self.portfolio_value = 10000
        self.positions = {}
        self.trades = []
        self.signals = []
    
    def run_backtest(self, data: pd.DataFrame, asset: str = "BTCUSDT") -> BacktestResult:
        """
        Run complete backtest
        
        Args:
            data: OHLCV data
            asset: Asset name
        
        Returns:
            BacktestResult object
        """
        print(f"🚀 Starting backtest for {asset}")
        print(f"📊 Data range: {data.index[0]} to {data.index[-1]}")
        print(f"📈 Data points: {len(data)}")
        
        # Reset state
        self.portfolio_value = 10000
        self.positions = {}
        self.trades = []
        self.signals = []
        
        # Build features
        print("🔧 Building features...")
        feature_df, feature_cols = self.feature_engine.build_feature_matrix(
            data, self.config.horizons_hours
        )
        
        # Add macro features
        feature_df = self.macro_features.calculate_macro_features(feature_df)
        
        print(f"✅ Features built: {len(feature_cols)} features")
        
        # Walk-forward backtest
        print("🔄 Running walk-forward backtest...")
        results = self._walk_forward_backtest(feature_df, asset)
        
        # Calculate performance metrics
        print("📊 Calculating performance metrics...")
        performance_metrics = self._calculate_performance_metrics()
        
        # Calculate model metrics
        model_metrics = self._calculate_model_metrics()
        
        # Analyze regimes
        regime_analysis = self._analyze_regimes()
        
        # Create result
        result = BacktestResult(
            trades=self.trades,
            signals=self.signals,
            performance_metrics=performance_metrics,
            model_metrics=model_metrics,
            regime_analysis=regime_analysis,
            timestamp=datetime.now()
        )
        
        print("✅ Backtest completed!")
        self._print_summary(result)
        
        return result
    
    def _walk_forward_backtest(self, feature_df: pd.DataFrame, asset: str) -> None:
        """
        Run walk-forward backtest
        
        Args:
            feature_df: Feature DataFrame
            asset: Asset name
        """
        train_days = self.config.train_days
        test_days = self.config.test_days
        embargo_hours = self.config.embargo_hours
        
        # Convert days to hours
        train_hours = train_days * 24
        test_hours = test_days * 24
        
        # Calculate step size
        step_hours = test_hours + embargo_hours
        
        # Get feature columns (exclude targets)
        feature_cols = [col for col in feature_df.columns if not col.startswith('target')]
        target_cols = [col for col in feature_df.columns if col.startswith('target')]
        
        # Initialize model
        model = None
        
        # Walk forward
        for i in range(train_hours, len(feature_df) - test_hours, step_hours):
            # Training period
            train_start = i - train_hours
            train_end = i
            train_data = feature_df.iloc[train_start:train_end]
            
            # Test period
            test_start = i + embargo_hours
            test_end = min(i + embargo_hours + test_hours, len(feature_df))
            test_data = feature_df.iloc[test_start:test_end]
            
            if len(train_data) < 100 or len(test_data) < 10:
                continue
            
            # Train model
            print(f"📚 Training model on {len(train_data)} samples...")
            try:
                X_train = train_data[feature_cols]
                y_train = train_data[target_cols[0]]  # Use first target
                
                model = self.model_trainer.train_model(
                    X_train, y_train, self.config.learner
                )
                print(f"✅ Model trained: {model.model_type}")
                
            except Exception as e:
                print(f"❌ Model training failed: {e}")
                continue
            
            # Test period
            print(f"🧪 Testing on {len(test_data)} samples...")
            self._test_period(model, test_data, feature_cols, target_cols[0], asset)
    
    def _test_period(self, model: TrainedModel, test_data: pd.DataFrame, 
                    feature_cols: List[str], target_col: str, asset: str) -> None:
        """
        Test period in walk-forward backtest
        
        Args:
            model: Trained model
            test_data: Test data
            feature_cols: Feature columns
            target_col: Target column
            asset: Asset name
        """
        for idx, row in test_data.iterrows():
            # Get features for current timestamp
            X_current = pd.DataFrame([row[feature_cols]])
            
            # Get predictions
            try:
                predictions, probabilities = self.model_trainer.predict(model, X_current)
                prob_up = probabilities[0, 1]
                prob_down = probabilities[0, 0]
            except Exception as e:
                print(f"❌ Prediction failed: {e}")
                continue
            
            # Get current market data
            current_price = row['close']
            volatility = row.get('volatility_24h', 0.02)
            
            # Detect regime
            regime_state = self.regime_detector.detect_regime(
                test_data.loc[:idx] if idx in test_data.index else test_data
            )
            regime_implications = self.regime_detector.get_regime_implications(regime_state)
            
            # Determine signal
            signal, confidence, metadata = self.threshold_manager.determine_signal(
                probabilities, volatility
            )
            
            # Log signal
            self.trading_logger.log_signal(
                signal=signal,
                confidence=confidence,
                prob_up=prob_up,
                prob_down=prob_down,
                asset=asset,
                regime=regime_state.regime.value,
                volatility=volatility,
                metadata=metadata
            )
            
            # Create trade plan if signal is valid
            if signal in ["LONG", "SHORT"]:
                trade_plan = self.trading_planner.create_trade_plan(
                    signal=signal,
                    entry_price=current_price,
                    confidence=confidence,
                    volatility=volatility,
                    regime_implications=regime_implications,
                    portfolio_value=self.portfolio_value
                )
                
                if trade_plan:
                    # Validate trade plan
                    is_valid, reason = self.trading_planner.validate_trade_plan(trade_plan)
                    
                    if is_valid:
                        # Calculate leverage
                        leverage = self.leverage_manager.calculate_optimal_leverage(
                            volatility=volatility,
                            confidence=confidence,
                            risk_reward=trade_plan.risk_reward,
                            regime_implications=regime_implications,
                            portfolio_value=self.portfolio_value
                        )
                        
                        # Adjust position size for leverage
                        final_position_size = self.leverage_manager.adjust_position_size_for_leverage(
                            trade_plan.position_size, leverage, self.portfolio_value
                        )
                        
                        # Log trade
                        trade_id = self.trading_logger.log_trade(
                            signal=signal,
                            entry_price=current_price,
                            stop_loss=trade_plan.stop_loss,
                            take_profit=trade_plan.take_profit,
                            position_size=final_position_size,
                            leverage=leverage,
                            confidence=confidence,
                            risk_reward=trade_plan.risk_reward,
                            asset=asset,
                            metadata={
                                'regime': regime_state.regime.value,
                                'volatility': volatility,
                                'timestamp': idx
                            }
                        )
                        
                        # Store trade
                        self.trades.append({
                            'trade_id': trade_id,
                            'timestamp': idx,
                            'signal': signal,
                            'entry_price': current_price,
                            'stop_loss': trade_plan.stop_loss,
                            'take_profit': trade_plan.take_profit,
                            'position_size': final_position_size,
                            'leverage': leverage,
                            'confidence': confidence,
                            'risk_reward': trade_plan.risk_reward,
                            'asset': asset
                        })
            
            # Store signal
            self.signals.append({
                'timestamp': idx,
                'signal': signal,
                'confidence': confidence,
                'prob_up': prob_up,
                'prob_down': prob_down,
                'asset': asset,
                'regime': regime_state.regime.value,
                'volatility': volatility
            })
    
    def _calculate_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.trades:
            return {}
        
        # Calculate returns
        returns = []
        for trade in self.trades:
            # Simplified P&L calculation (assuming all trades closed at take profit or stop loss)
            if trade['signal'] == 'LONG':
                # Assume 50% hit take profit, 50% hit stop loss
                pnl = (trade['take_profit'] - trade['entry_price']) * trade['position_size'] * 0.5 + \
                      (trade['stop_loss'] - trade['entry_price']) * trade['position_size'] * 0.5
            else:  # SHORT
                pnl = (trade['entry_price'] - trade['take_profit']) * trade['position_size'] * 0.5 + \
                      (trade['entry_price'] - trade['stop_loss']) * trade['position_size'] * 0.5
            
            returns.append(pnl / self.portfolio_value)
        
        # Calculate metrics
        total_return = sum(returns)
        avg_return = np.mean(returns) if returns else 0
        return_std = np.std(returns) if returns else 0
        
        # Sharpe ratio (simplified)
        sharpe_ratio = avg_return / return_std if return_std > 0 else 0
        
        # Win rate
        win_rate = len([r for r in returns if r > 0]) / len(returns) if returns else 0
        
        # Maximum drawdown (simplified)
        cumulative_returns = np.cumsum(returns)
        max_drawdown = np.min(cumulative_returns - np.maximum.accumulate(cumulative_returns))
        
        return {
            'total_return': total_return,
            'avg_return': avg_return,
            'return_std': return_std,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'avg_confidence': np.mean([t['confidence'] for t in self.trades]),
            'avg_risk_reward': np.mean([t['risk_reward'] for t in self.trades])
        }
    
    def _calculate_model_metrics(self) -> Dict:
        """
        Calculate model metrics
        
        Returns:
            Dictionary with model metrics
        """
        if not self.signals:
            return {}
        
        # Signal distribution
        signal_counts = {}
        for signal in self.signals:
            signal_counts[signal['signal']] = signal_counts.get(signal['signal'], 0) + 1
        
        # Average confidence by signal type
        confidence_by_signal = {}
        for signal_type in ['LONG', 'SHORT', 'HOLD']:
            signal_confidences = [s['confidence'] for s in self.signals if s['signal'] == signal_type]
            if signal_confidences:
                confidence_by_signal[signal_type] = np.mean(signal_confidences)
        
        return {
            'total_signals': len(self.signals),
            'signal_distribution': signal_counts,
            'avg_confidence_by_signal': confidence_by_signal,
            'avg_confidence': np.mean([s['confidence'] for s in self.signals]),
            'avg_volatility': np.mean([s['volatility'] for s in self.signals])
        }
    
    def _analyze_regimes(self) -> Dict:
        """
        Analyze market regimes
        
        Returns:
            Dictionary with regime analysis
        """
        if not self.signals:
            return {}
        
        # Regime distribution
        regime_counts = {}
        for signal in self.signals:
            regime = signal['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Performance by regime
        regime_performance = {}
        for regime in regime_counts.keys():
            regime_signals = [s for s in self.signals if s['regime'] == regime]
            if regime_signals:
                regime_performance[regime] = {
                    'count': len(regime_signals),
                    'avg_confidence': np.mean([s['confidence'] for s in regime_signals]),
                    'signal_distribution': {}
                }
                
                for signal in regime_signals:
                    signal_type = signal['signal']
                    regime_performance[regime]['signal_distribution'][signal_type] = \
                        regime_performance[regime]['signal_distribution'].get(signal_type, 0) + 1
        
        return {
            'regime_distribution': regime_counts,
            'regime_performance': regime_performance
        }
    
    def _print_summary(self, result: BacktestResult) -> None:
        """
        Print backtest summary
        
        Args:
            result: Backtest result
        """
        print("\n" + "="*50)
        print("📊 BACKTEST SUMMARY")
        print("="*50)
        
        # Performance metrics
        perf = result.performance_metrics
        if perf:
            print(f"💰 Total Return: {perf['total_return']:.2%}")
            print(f"📈 Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
            print(f"🎯 Win Rate: {perf['win_rate']:.2%}")
            print(f"📉 Max Drawdown: {perf['max_drawdown']:.2%}")
            print(f"🔄 Total Trades: {perf['total_trades']}")
        
        # Model metrics
        model = result.model_metrics
        if model:
            print(f"\n🤖 MODEL METRICS")
            print(f"📡 Total Signals: {model['total_signals']}")
            print(f"📊 Signal Distribution: {model['signal_distribution']}")
            print(f"🎯 Avg Confidence: {model['avg_confidence']:.2f}")
        
        # Regime analysis
        regime = result.regime_analysis
        if regime:
            print(f"\n🌍 REGIME ANALYSIS")
            print(f"📊 Regime Distribution: {regime['regime_distribution']}")
        
        print("="*50)
