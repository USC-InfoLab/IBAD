"""Repo-root anchor and YAML config loader."""

from pathlib import Path
import yaml

# src/utils/config.py → parents: [0]=utils [1]=src [2]=IBAD root
ROOT = Path(__file__).resolve().parents[2]


def load_cfg(config_path: str) -> dict:
    """Load a YAML config file at *config_path* relative to the repo root."""
    return yaml.safe_load((ROOT / config_path).read_text())
