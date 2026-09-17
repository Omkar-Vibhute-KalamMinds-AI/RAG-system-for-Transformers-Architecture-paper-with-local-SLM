from pathlib import Path
import os
import yaml


_project_root = Path(__file__).resolve().parent.parent


def load_config(config_name: str = "config.yaml") -> dict:
    """Load a YAML config from the project root."""
    config_path = _project_root / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config


if __name__ == "__main__":
    print(_project_root)
    print(load_config())
