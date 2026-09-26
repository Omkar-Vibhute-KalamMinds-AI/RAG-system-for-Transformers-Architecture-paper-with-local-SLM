from pathlib import Path
import os
import yaml


_project_root = Path(__file__).resolve().parent.parent

_ENV_OVERRIDES = (
    ("GEMMA_MODEL_PATH", ("llm", "local_path")),
    ("EMBED_MODEL_PATH", ("embedding model", "local_path")),
    ("LANCEDB_PATH", ("vectordb", "path")),
    ("LANCEDB_TABLE", ("vectordb", "table")),
)

def _config_path(config_name: str | None) -> Path:
    if config_name is None:
        config_name = os.environ.get("LLMOPS_CONFIG") or os.environ.get("CONFIG_PATH") or "config.yaml"
    path = Path(config_name)
    if path.is_absolute():
        return path
    return _project_root / path


def _apply_env_overrides(config: dict) -> dict:
    for env_name, keys in _ENV_OVERRIDES:
        value = os.environ.get(env_name)
        if not value:
            continue
        section, key = keys
        if section not in config or not isinstance(config[section], dict):
            config[section] = {}
        config[section][key] = value
    return config


def load_config(config_name: str | None = None) -> dict:
    """Load YAML from the project root, then overlay path env vars if set."""
    config_path = _config_path(config_name)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return _apply_env_overrides(config)

if __name__ == "__main__":
    print(_project_root)
    print(load_config())
