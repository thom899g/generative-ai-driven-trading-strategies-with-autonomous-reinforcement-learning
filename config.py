"""
Configuration management for the Generative AI Trading System.
Centralized config with environment variables and defaults.
"""
import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class FirebaseConfig:
    """Firebase configuration for state management"""
    credential_path: str = field(
        default_factory=lambda: os.getenv('FIREBASE_CREDENTIAL_PATH', './firebase_credentials.json')
    )
    project_id: str = field(
        default_factory=lambda: os.getenv('FIREBASE_PROJECT_ID', 'generative-trading-system')
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv('FIRESTORE_COLLECTION', 'trading_states')
    )
    realtime_db_url: str = field(
        default_factory=lambda: os.getenv('FIREBASE_REALTIME_DB_URL', '')
    )

@dataclass
class TradingConfig:
    """Trading-specific configuration"""
    symbols: List[str] = field(default_factory=lambda: ['BTC/USDT', 'ETH/USDT', 'AAPL'])
    timeframe: str = '1h'
    initial_capital: float = 100000.0
    max_position_size: float = 0.1  # 10% of portfolio
    risk_per_trade: float = 0.02  # 2% risk per trade
    commission_rate: float = 0.001  # 0.1% commission

@dataclass
class ModelConfig:
    """AI/ML model configuration"""
    # Generative Model
    gen_model_path: str = './models/generative_model'
    gen_latent_dim: int = 128
    gen_hidden_layers: List[int] = field(default_factory=lambda: [256, 512, 256])
    
    # RL Agent
    rl_learning_rate: float = 0.001
    rl_gamma: float = 0.99
    rl_memory_size: int = 10000
    rl_batch_size: int = 64
    rl_update_frequency: int = 100
    
    # Unsupervised Learning
    n_clusters: int = 10
    anomaly_threshold: float = 2.5

@dataclass
class SystemConfig:
    """System and infrastructure configuration"""
    log_level: str = field(
        default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO')
    )
    log_file: str = './logs/trading_system.log'
    max_retries: int = 3
    retry_delay: int = 5
    simulation_mode: bool = field(
        default_factory=lambda: os.getenv('SIMULATION_MODE', 'True').lower() == 'true'
    )
    data_cache_dir: str = './data/cache'
    model_checkpoint_dir: str = './checkpoints'
    
    # Monitoring
    prometheus_port: int = 8000
    health_check_interval: int = 60

@dataclass
class Config:
    """Main configuration container"""
    firebase: FirebaseConfig = field(default_factory=FirebaseConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    # Telegram alerts (for emergency contact)
    telegram_bot_token: Optional[str] = field(
        default_factory=lambda: os.getenv('TELEGRAM_BOT_TOKEN')
    )
    telegram_chat_id: Optional[str] = field(
        default_factory=lambda: os.getenv('TELEGRAM_CHAT_ID')
    )
    
    def __post_init__(self):
        """Initialize directories and validate config"""
        # Create necessary directories
        Path(self.system.log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.system.data_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.system.model_checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate critical paths
        if not self.firebase.credential_path or not Path(self.firebase.credential_path).exists():
            logging.warning(f"Firebase credential file not found: {self.firebase.credential_path}")
        
        # Validate simulation mode
        if not self.system.simulation_mode:
            logging.warning("LIVE TRADING MODE ENABLED - Proceed with caution")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return {
            'trading': self.trading.__dict__,
            'model': self.model.__dict__,
            'system': self.system.__dict__,
            'firebase': self.firebase.__dict__
        }

# Global config instance
config = Config()

# Initialize logging
logging.basicConfig(
    level=getattr(logging, config.system.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.system.log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)