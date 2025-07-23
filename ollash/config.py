import yaml
from pathlib import Path
from importlib.resources import files
from platformdirs import user_config_dir

# Constants
APP_NAME = "ollash"
PACKAGE_NAME = "ollash"
PACKAGE_CONFIG_NAME = "config.yaml"

USER_CONFIG_PATH = Path(user_config_dir(APP_NAME)) / "config.yaml"

def _copy_default_config():
    """Copy default config to user config directory if not already present."""
    if USER_CONFIG_PATH.exists():
        return

    USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    default_text = files(PACKAGE_NAME).joinpath(PACKAGE_CONFIG_NAME).read_text()
    USER_CONFIG_PATH.write_text(default_text)

def load_config():
    """
    Ensures config exists at the user config directory,
    then loads and returns it as a dictionary.
    """
    _copy_default_config()
    with open(USER_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def get_config_value(key: str, default=None):
    """
    Safely get a value from the config file, with optional default.
    """
    config = load_config()
    return config.get(key, default)

def get_config_path():
    """Return the full path to the user config file."""
    return USER_CONFIG_PATH
