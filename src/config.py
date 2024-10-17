import yaml
from typing import Dict, Any


class Config:
    """Configuration class to load, save and update configuration"""

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    

    @staticmethod
    def save_config(config: Dict, save_path: str) -> None:
        """Save configuration to YAML file"""
        with open(save_path, 'w') as f:
            yaml.dump(config, f)


    @staticmethod
    def update_config(config: Dict, updates: Dict) -> Dict:
        """Update configuration with new parameters"""
        for key, value in updates.items():
            if '.' in key:
                # Handle nested updates like 'training.learning_rate'
                keys = key.split('.')
                conf = config
                for k in keys[:-1]:
                    conf = conf[k]
                conf[keys[-1]] = value
            else:
                config[key] = value
        return config