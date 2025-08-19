#!/usr/bin/env python3
"""
Optimizer for alpha12_24
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ..backtest.runner import BacktestRunner
from ..core.config import config


@dataclass
class OptimizationResult:
    """Optimization result structure"""
    best_params: Dict
    best_score: float
    all_results: List[Dict]
    optimization_history: List[Dict]
    timestamp: datetime


class HyperparameterOptimizer:
    """Hyperparameter optimizer for alpha12_24"""
    
    def __init__(self, config):
        self.config = config
        self.backtest_runner = BacktestRunner(config)
    
    def optimize_model_hyperparameters(self, data: pd.DataFrame, 
                                     param_grid: Dict, 
                                     metric: str = 'sharpe_ratio',
                                     n_trials: int = 20) -> OptimizationResult:
        """
        Optimize model hyperparameters
        
        Args:
            data: OHLCV data
            param_grid: Parameter grid for optimization
            metric: Optimization metric
            n_trials: Number of optimization trials
        
        Returns:
            OptimizationResult object
        """
        print(f"🔧 Starting hyperparameter optimization with {n_trials} trials")
        print(f"📊 Optimization metric: {metric}")
        
        best_score = -np.inf
        best_params = None
        all_results = []
        optimization_history = []
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid, n_trials)
        
        for i, params in enumerate(param_combinations):
            print(f"🔄 Trial {i+1}/{len(param_combinations)}")
            print(f"📋 Parameters: {params}")
            
            try:
                # Update config with new parameters
                temp_config = self._create_temp_config(params)
                
                # Run backtest with new parameters
                result = self._run_backtest_with_params(data, temp_config)
                
                # Extract score
                score = self._extract_score(result, metric)
                
                # Store result
                trial_result = {
                    'trial': i + 1,
                    'params': params,
                    'score': score,
                    'performance_metrics': result.performance_metrics,
                    'model_metrics': result.model_metrics
                }
                all_results.append(trial_result)
                optimization_history.append({
                    'trial': i + 1,
                    'score': score,
                    'timestamp': datetime.now()
                })
                
                # Update best result
                if score > best_score:
                    best_score = score
                    best_params = params
                    print(f"✅ New best score: {best_score:.4f}")
                
                print(f"📊 Score: {score:.4f}")
                
            except Exception as e:
                print(f"❌ Trial {i+1} failed: {e}")
                continue
        
        print(f"🎯 Optimization completed!")
        print(f"🏆 Best score: {best_score:.4f}")
        print(f"🔧 Best parameters: {best_params}")
        
        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=all_results,
            optimization_history=optimization_history,
            timestamp=datetime.now()
        )
    
    def optimize_thresholds(self, data: pd.DataFrame,
                          threshold_ranges: Dict,
                          metric: str = 'sharpe_ratio',
                          n_trials: int = 30) -> OptimizationResult:
        """
        Optimize signal thresholds
        
        Args:
            data: OHLCV data
            threshold_ranges: Threshold ranges for optimization
            metric: Optimization metric
            n_trials: Number of optimization trials
        
        Returns:
            OptimizationResult object
        """
        print(f"🎯 Starting threshold optimization with {n_trials} trials")
        print(f"📊 Optimization metric: {metric}")
        
        best_score = -np.inf
        best_params = None
        all_results = []
        optimization_history = []
        
        # Generate threshold combinations
        threshold_combinations = self._generate_threshold_combinations(threshold_ranges, n_trials)
        
        for i, thresholds in enumerate(threshold_combinations):
            print(f"🔄 Trial {i+1}/{len(threshold_combinations)}")
            print(f"📋 Thresholds: {thresholds}")
            
            try:
                # Update config with new thresholds
                temp_config = self._create_temp_config(thresholds)
                
                # Run backtest with new thresholds
                result = self._run_backtest_with_params(data, temp_config)
                
                # Extract score
                score = self._extract_score(result, metric)
                
                # Store result
                trial_result = {
                    'trial': i + 1,
                    'thresholds': thresholds,
                    'score': score,
                    'performance_metrics': result.performance_metrics,
                    'model_metrics': result.model_metrics
                }
                all_results.append(trial_result)
                optimization_history.append({
                    'trial': i + 1,
                    'score': score,
                    'timestamp': datetime.now()
                })
                
                # Update best result
                if score > best_score:
                    best_score = score
                    best_params = thresholds
                    print(f"✅ New best score: {best_score:.4f}")
                
                print(f"📊 Score: {score:.4f}")
                
            except Exception as e:
                print(f"❌ Trial {i+1} failed: {e}")
                continue
        
        print(f"🎯 Threshold optimization completed!")
        print(f"🏆 Best score: {best_score:.4f}")
        print(f"🔧 Best thresholds: {best_params}")
        
        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=all_results,
            optimization_history=optimization_history,
            timestamp=datetime.now()
        )
    
    def optimize_risk_parameters(self, data: pd.DataFrame,
                               risk_ranges: Dict,
                               metric: str = 'sharpe_ratio',
                               n_trials: int = 25) -> OptimizationResult:
        """
        Optimize risk management parameters
        
        Args:
            data: OHLCV data
            risk_ranges: Risk parameter ranges
            metric: Optimization metric
            n_trials: Number of optimization trials
        
        Returns:
            OptimizationResult object
        """
        print(f"🛡️ Starting risk parameter optimization with {n_trials} trials")
        print(f"📊 Optimization metric: {metric}")
        
        best_score = -np.inf
        best_params = None
        all_results = []
        optimization_history = []
        
        # Generate risk parameter combinations
        risk_combinations = self._generate_risk_combinations(risk_ranges, n_trials)
        
        for i, risk_params in enumerate(risk_combinations):
            print(f"🔄 Trial {i+1}/{len(risk_combinations)}")
            print(f"📋 Risk parameters: {risk_params}")
            
            try:
                # Update config with new risk parameters
                temp_config = self._create_temp_config(risk_params)
                
                # Run backtest with new risk parameters
                result = self._run_backtest_with_params(data, temp_config)
                
                # Extract score
                score = self._extract_score(result, metric)
                
                # Store result
                trial_result = {
                    'trial': i + 1,
                    'risk_params': risk_params,
                    'score': score,
                    'performance_metrics': result.performance_metrics,
                    'model_metrics': result.model_metrics
                }
                all_results.append(trial_result)
                optimization_history.append({
                    'trial': i + 1,
                    'score': score,
                    'timestamp': datetime.now()
                })
                
                # Update best result
                if score > best_score:
                    best_score = score
                    best_params = risk_params
                    print(f"✅ New best score: {best_score:.4f}")
                
                print(f"📊 Score: {score:.4f}")
                
            except Exception as e:
                print(f"❌ Trial {i+1} failed: {e}")
                continue
        
        print(f"🎯 Risk parameter optimization completed!")
        print(f"🏆 Best score: {best_score:.4f}")
        print(f"🔧 Best risk parameters: {best_params}")
        
        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=all_results,
            optimization_history=optimization_history,
            timestamp=datetime.now()
        )
    
    def _generate_param_combinations(self, param_grid: Dict, n_trials: int) -> List[Dict]:
        """
        Generate parameter combinations for optimization
        
        Args:
            param_grid: Parameter grid
            n_trials: Number of trials
        
        Returns:
            List of parameter combinations
        """
        combinations = []
        
        for _ in range(n_trials):
            params = {}
            for param_name, param_range in param_grid.items():
                if isinstance(param_range, list):
                    params[param_name] = np.random.choice(param_range)
                elif isinstance(param_range, tuple) and len(param_range) == 2:
                    if isinstance(param_range[0], int):
                        params[param_name] = np.random.randint(param_range[0], param_range[1])
                    else:
                        params[param_name] = np.random.uniform(param_range[0], param_range[1])
                else:
                    params[param_name] = param_range
            
            combinations.append(params)
        
        return combinations
    
    def _generate_threshold_combinations(self, threshold_ranges: Dict, n_trials: int) -> List[Dict]:
        """
        Generate threshold combinations
        
        Args:
            threshold_ranges: Threshold ranges
            n_trials: Number of trials
        
        Returns:
            List of threshold combinations
        """
        combinations = []
        
        for _ in range(n_trials):
            thresholds = {}
            for threshold_name, threshold_range in threshold_ranges.items():
                if isinstance(threshold_range, tuple) and len(threshold_range) == 2:
                    thresholds[threshold_name] = np.random.uniform(threshold_range[0], threshold_range[1])
                else:
                    thresholds[threshold_name] = threshold_range
            
            combinations.append(thresholds)
        
        return combinations
    
    def _generate_risk_combinations(self, risk_ranges: Dict, n_trials: int) -> List[Dict]:
        """
        Generate risk parameter combinations
        
        Args:
            risk_ranges: Risk parameter ranges
            n_trials: Number of trials
        
        Returns:
            List of risk parameter combinations
        """
        combinations = []
        
        for _ in range(n_trials):
            risk_params = {}
            for param_name, param_range in risk_ranges.items():
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    risk_params[param_name] = np.random.uniform(param_range[0], param_range[1])
                else:
                    risk_params[param_name] = param_range
            
            combinations.append(risk_params)
        
        return combinations
    
    def _create_temp_config(self, params: Dict) -> Any:
        """
        Create temporary config with new parameters
        
        Args:
            params: New parameters
        
        Returns:
            Temporary config object
        """
        # Create a copy of the config
        temp_config = type('TempConfig', (), {})()
        
        # Copy all attributes from original config
        for attr in dir(self.config):
            if not attr.startswith('_'):
                setattr(temp_config, attr, getattr(self.config, attr))
        
        # Update with new parameters
        for param_name, param_value in params.items():
            if hasattr(temp_config, param_name):
                setattr(temp_config, param_name, param_value)
            else:
                # Handle nested parameters
                if '.' in param_name:
                    section, key = param_name.split('.', 1)
                    if not hasattr(temp_config, section):
                        setattr(temp_config, section, {})
                    getattr(temp_config, section)[key] = param_value
        
        return temp_config
    
    def _run_backtest_with_params(self, data: pd.DataFrame, temp_config: Any) -> Any:
        """
        Run backtest with temporary parameters
        
        Args:
            data: OHLCV data
            temp_config: Temporary config
        
        Returns:
            Backtest result
        """
        # Create temporary backtest runner
        temp_runner = BacktestRunner(temp_config)
        
        # Run backtest
        result = temp_runner.run_backtest(data)
        
        return result
    
    def _extract_score(self, result: Any, metric: str) -> float:
        """
        Extract score from backtest result
        
        Args:
            result: Backtest result
            metric: Metric name
        
        Returns:
            Score value
        """
        if not result.performance_metrics:
            return -np.inf
        
        if metric == 'sharpe_ratio':
            return result.performance_metrics.get('sharpe_ratio', -np.inf)
        elif metric == 'total_return':
            return result.performance_metrics.get('total_return', -np.inf)
        elif metric == 'win_rate':
            return result.performance_metrics.get('win_rate', -np.inf)
        elif metric == 'max_drawdown':
            return -abs(result.performance_metrics.get('max_drawdown', 0))  # Minimize drawdown
        elif metric == 'calmar_ratio':
            total_return = result.performance_metrics.get('total_return', 0)
            max_dd = abs(result.performance_metrics.get('max_drawdown', 1))
            return total_return / max_dd if max_dd > 0 else -np.inf
        else:
            return result.performance_metrics.get(metric, -np.inf)
    
    def get_optimization_summary(self, result: OptimizationResult) -> Dict:
        """
        Get optimization summary
        
        Args:
            result: Optimization result
        
        Returns:
            Dictionary with optimization summary
        """
        if not result.all_results:
            return {}
        
        scores = [r['score'] for r in result.all_results]
        
        summary = {
            'best_score': result.best_score,
            'best_params': result.best_params,
            'total_trials': len(result.all_results),
            'avg_score': np.mean(scores),
            'score_std': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'successful_trials': len([s for s in scores if s > -np.inf])
        }
        
        return summary
    
    def export_optimization_results(self, result: OptimizationResult, 
                                  filename: str = None) -> str:
        """
        Export optimization results to CSV
        
        Args:
            result: Optimization result
            filename: Output filename (optional)
        
        Returns:
            Path to exported file
        """
        if not result.all_results:
            return ""
        
        if filename is None:
            filename = f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Convert results to DataFrame
        results_data = []
        for trial_result in result.all_results:
            row = {
                'trial': trial_result['trial'],
                'score': trial_result['score']
            }
            
            # Add parameters
            if 'params' in trial_result:
                row.update(trial_result['params'])
            elif 'thresholds' in trial_result:
                row.update(trial_result['thresholds'])
            elif 'risk_params' in trial_result:
                row.update(trial_result['risk_params'])
            
            # Add performance metrics
            perf_metrics = trial_result.get('performance_metrics', {})
            for metric, value in perf_metrics.items():
                row[f'perf_{metric}'] = value
            
            results_data.append(row)
        
        df = pd.DataFrame(results_data)
        df.to_csv(filename, index=False)
        
        return filename
