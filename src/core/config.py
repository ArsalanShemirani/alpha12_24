"""
Configuration management for alpha12_24
"""

import yaml
import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration class for alpha12_24"""
    
    # Hard defaults used across UI/daemon
    runs_dir: str = os.getenv("ALPHA12_RUNS_DIR", "runs")
    model_dir: str = os.getenv("ALPHA12_MODEL_DIR", "artifacts")
    bar_interval: str = os.getenv("ALPHA12_INTERVAL", "15m")
    risk_per_trade: float = float(os.getenv("RISK_PER_TRADE", "1.0"))  # percent
    stop_min_frac: float = float(os.getenv("STOP_MIN_FRAC", "0.005"))   # 0.5%
    min_rr: float = float(os.getenv("MIN_RR", "1.8"))
    taker_bps_per_side: float = float(os.getenv("TAKER_BPS_PER_SIDE", "5"))
    assets: list[str] = os.getenv("ALPHA12_ASSETS", "BTCUSDT,ETHUSDT").split(",")
    horizons_hours: list[int] = [1]   # ensure exists; UI overrides as needed
    learner: str = os.getenv("LEARNER", "rf")
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration from YAML file"""
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            # Create default config if file doesn't exist
            self._create_default_config()
        
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _create_default_config(self):
        """Create default configuration file"""
        default_config = {
            'project': {
                'assets': ['BTCUSDT'],
                'horizons_hours': [12, 24],
                'bar_interval': '5m'
            },
            'model': {
                'learner': 'xgb',
                'calibrate': True,
                'train_days': 90,
                'test_days': 14,
                'embargo_hours': 24
            },
            'signal': {
                'prob_long': 0.60,
                'prob_short': 0.40,
                'min_rr': 1.8
            },
            'risk': {
                'risk_per_trade': 0.01,
                'stop_min_frac': 0.003
            },
            'fees': {
                'taker_bps_per_side': 7.5
            },
            'eval': {
                'horizon_choice_window_days': 60,
                'trades_per_month_cap': 60
            }
        }
        
        with open(self.config_path, 'w') as file:
            yaml.dump(default_config, file, default_flow_style=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports nested keys with dots)"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    @property
    def assets(self) -> List[str]:
        """Get list of assets"""
        return self.get('project.assets', ['BTCUSDT'])
    
    @property
    def horizons_hours(self) -> List[int]:
        """Get list of horizons in hours"""
        return self.get('project.horizons_hours', [12, 24])
    
    @property
    def bar_interval(self) -> str:
        """Get bar interval"""
        return self.get('project.bar_interval', '5m')
    
    @property
    def learner(self) -> str:
        """Get model learner type"""
        return self.get('model.learner', 'xgb')
    
    @property
    def calibrate(self) -> bool:
        """Get calibration flag"""
        return self.get('model.calibrate', True)
    
    @property
    def train_days(self) -> int:
        """Get training days"""
        return self.get('model.train_days', 90)
    
    @property
    def test_days(self) -> int:
        """Get test days"""
        return self.get('model.test_days', 14)
    
    @property
    def embargo_hours(self) -> int:
        """Get embargo hours"""
        return self.get('model.embargo_hours', 24)
    
    @property
    def prob_long(self) -> float:
        """Get long probability threshold"""
        return self.get('signal.prob_long', 0.60)
    
    @property
    def prob_short(self) -> float:
        """Get short probability threshold"""
        return self.get('signal.prob_short', 0.40)
    
    @property
    def min_rr(self) -> float:
        """Get minimum risk/reward ratio"""
        return self.get('signal.min_rr', 1.8)
    
    @property
    def risk_per_trade(self) -> float:
        """Get risk per trade"""
        return self.get('risk.risk_per_trade', 0.01)
    
    @property
    def stop_min_frac(self) -> float:
        """Get minimum stop loss fraction"""
        return self.get('risk.stop_min_frac', 0.003)
    
    @property
    def taker_bps_per_side(self) -> float:
        """Get taker fees in basis points"""
        return self.get('fees.taker_bps_per_side', 7.5)
    
    @property
    def horizon_choice_window_days(self) -> int:
        """Get horizon choice window days"""
        return self.get('eval.horizon_choice_window_days', 60)
    
    @property
    def trades_per_month_cap(self) -> int:
        """Get trades per month cap"""
        return self.get('eval.trades_per_month_cap', 60)

# Create global config instance
config = Config()
