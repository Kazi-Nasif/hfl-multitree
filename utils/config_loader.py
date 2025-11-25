"""
Configuration loader utility for HFL MultiTree project
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to YAML config file
        """
        if config_path is None:
            # Default to system_config.yaml in config directory
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "system_config.yaml"
        
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def get(self, key: str, default=None):
        """
        Get configuration value by key
        
        Args:
            key: Configuration key (supports nested keys with dots)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def __getitem__(self, key):
        """Allow dictionary-style access"""
        return self.get(key)
    
    def __repr__(self):
        """String representation"""
        return f"Config(path={self.config_path})"
    
    def print_config(self):
        """Pretty print configuration"""
        import json
        print(json.dumps(self.config, indent=2))


# Test function
if __name__ == "__main__":
    # Test loading config
    config = Config()
    
    print("=" * 50)
    print("Testing Configuration Loader")
    print("=" * 50)
    
    # Test accessing nested values
    print(f"\nTopology type: {config.get('topology.type')}")
    print(f"Number of nodes: {config.get('topology.num_nodes')}")
    print(f"Number of GPUs: {config.get('hardware.num_gpus')}")
    print(f"MultiTree k-ary: {config.get('multitree.k_ary')}")
    print(f"Time budget: {config.get('resources.time_budget_seconds')} seconds")
    
    print("\n" + "=" * 50)
    print("Full Configuration:")
    print("=" * 50)
    config.print_config()
