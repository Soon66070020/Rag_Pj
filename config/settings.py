"""Centralized configuration management for the RAG system.

This module loads and validates all configuration settings from:
- Environment variables (.env file)
- YAML configuration files (model_config.yaml, prompts.yaml)
- JSON schema files (weaviate_schema.json)

All settings are loaded once at startup and made available throughout
the application.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dotenv import load_dotenv

from src.core.exceptions import ConfigurationError


class Settings:
    """Global configuration settings for the RAG system.

    Loads configuration from multiple sources:
    - .env file: API keys, database URLs, paths
    - model_config.yaml: Model parameters and retrieval settings
    - prompts.yaml: Prompt templates
    - weaviate_schema.json: Database schema definition

    Attributes:
        deepseek_api_key: DeepSeek API key.
        deepseek_api_base: DeepSeek API base URL.
        weaviate_url: Weaviate database URL.
        weaviate_api_key: Weaviate API key (optional).
        project_root: Root directory of the project.
        config_dir: Configuration files directory.
        data_dir: Data directory (raw, processed, evaluation).
        logs_dir: Logs directory.
        outputs_dir: Outputs directory for evaluation results.
        model_config: Model configuration dictionary.
        prompts: Prompt templates dictionary.
        weaviate_schema: Weaviate schema definition.
    """

    def __init__(self, env_file: str = ".env"):
        """Initialize settings by loading all configuration sources.

        Args:
            env_file: Path to the .env file (relative to project root).

        Raises:
            ConfigurationError: If required settings are missing or invalid.
        """
        # Determine project root (parent of config directory)
        self.project_root = Path(__file__).parent.parent

        # Load environment variables from .env file
        env_path = self.project_root / env_file
        if env_path.exists():
            load_dotenv(env_path)

        # Load environment variables
        self._load_env_variables()

        # Set up directory paths
        self._setup_paths()

        # Load YAML and JSON configurations
        self._load_config_files()

        # Validate all settings
        self.validate()

    def _load_env_variables(self):
        """Load required environment variables."""
        # API Keys
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.deepseek_api_base = os.getenv(
            "DEEPSEEK_API_BASE",
            "https://api.deepseek.com/v1"
        )

        # Weaviate Database
        self.weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        self.weaviate_api_key = os.getenv("WEAVIATE_API_KEY")  # Optional

        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.enable_logging = os.getenv("ENABLE_LOGGING", "True").lower() == "true"

        # Model Cache
        self.hf_home = os.getenv("HF_HOME", str(self.project_root / "models_cache"))
        self.transformers_cache = os.getenv(
            "TRANSFORMERS_CACHE",
            str(self.project_root / "models_cache")
        )

        # Thai Language
        self.pythainlp_data_dir = os.getenv(
            "PYTHAINLP_DATA_DIR",
            str(self.project_root / "pythainlp_data")
        )

        # Debug mode
        self.debug = os.getenv("DEBUG", "False").lower() == "true"

    def _setup_paths(self):
        """Set up all directory paths."""
        self.config_dir = self.project_root / "config"
        self.data_dir = self.project_root / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.evaluation_data_dir = self.data_dir / "evaluation"
        self.logs_dir = self.project_root / "logs"
        self.outputs_dir = self.project_root / "outputs"

        # Create directories if they don't exist
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def _load_config_files(self):
        """Load YAML and JSON configuration files."""
        try:
            # Load model configuration
            self.model_config = self._load_yaml("model_config.yaml")

            # Load prompts
            self.prompts = self._load_yaml("prompts.yaml")

            # Load Weaviate schema
            self.weaviate_schema = self._load_json("weaviate_schema.json")

        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration files: {e}")

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file.

        Args:
            filename: Name of the YAML file in config directory.

        Returns:
            Dictionary containing the YAML contents.

        Raises:
            ConfigurationError: If file not found or invalid YAML.
        """
        filepath = self.config_dir / filename
        if not filepath.exists():
            raise ConfigurationError(f"Configuration file not found: {filepath}")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {filename}: {e}")

    def _load_json(self, filename: str) -> Dict[str, Any]:
        """Load a JSON configuration file.

        Args:
            filename: Name of the JSON file in config directory.

        Returns:
            Dictionary containing the JSON contents.

        Raises:
            ConfigurationError: If file not found or invalid JSON.
        """
        filepath = self.config_dir / filename
        if not filepath.exists():
            raise ConfigurationError(f"Configuration file not found: {filepath}")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in {filename}: {e}")

    def validate(self) -> bool:
        """Validate that all required settings are present and valid.

        Returns:
            True if validation succeeds.

        Raises:
            ConfigurationError: If validation fails.
        """
        # Check required API keys
        required_vars = [
            ("DEEPSEEK_API_KEY", self.deepseek_api_key, "DeepSeek API key is required"),
            ("WEAVIATE_URL", self.weaviate_url, "Weaviate URL is required"),
        ]

        for var_name, var_value, error_msg in required_vars:
            if not var_value:
                raise ConfigurationError(f"{error_msg} (set {var_name} in .env)")

        # Validate model config structure
        required_model_sections = ["embedding", "reranker", "llm", "retrieval"]
        for section in required_model_sections:
            if section not in self.model_config:
                raise ConfigurationError(
                    f"Missing '{section}' section in model_config.yaml"
                )

        # Validate prompts
        required_prompts = ["system_prompt", "hyde_prompt_template"]
        for prompt_key in required_prompts:
            if prompt_key not in self.prompts:
                raise ConfigurationError(
                    f"Missing '{prompt_key}' in prompts.yaml"
                )

        # Validate Weaviate schema
        if "class" not in self.weaviate_schema:
            raise ConfigurationError(
                "Missing 'class' field in weaviate_schema.json"
            )

        return True

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot-notation key.

        Supports nested keys like "model_config.embedding.model_name".

        Args:
            key: Configuration key (dot-separated for nested values).
            default: Default value if key not found.

        Returns:
            Configuration value or default.

        Example:
            >>> settings.get("model_config.embedding.model_name")
            'BAAI/bge-m3'
        """
        keys = key.split('.')
        value = self.__dict__

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                value = getattr(value, k, None)

            if value is None:
                return default

        return value

    def __repr__(self) -> str:
        """String representation of settings (without sensitive data)."""
        return (
            f"Settings("
            f"deepseek_api_key={'***' if self.deepseek_api_key else 'None'}, "
            f"weaviate_url={self.weaviate_url}, "
            f"debug={self.debug}"
            f")"
        )


# Global settings instance (loaded once)
_settings: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """Get the global settings instance.

    Args:
        reload: If True, reload settings from files.

    Returns:
        Settings instance.

    Example:
        >>> from config.settings import get_settings
        >>> settings = get_settings()
        >>> print(settings.deepseek_api_key)
    """
    global _settings

    if _settings is None or reload:
        _settings = Settings()

    return _settings
