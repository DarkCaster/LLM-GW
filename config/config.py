import yaml
import os

# Config class reflects the following configuration (example)

# models_dir: "/tmp"
#
# models:
#   - name: "qwen"
#     subdir: "qwen_tensors"
#   - name: "deepseek"
#     subdir: "deepseek_tensors"


class Config:
    def __init__(self, config_file_name):
        self._config_file_name = config_file_name
        self._config_data = self._load_config()
        self._validate_config(self._config_data)

        # Assigning properties based on the validated config data
        self.models_dir = self._config_data["models_dir"]
        self.models = self._config_data.get("models", [])

    def _load_config(self):
        if not os.path.exists(self._config_file_name):
            raise FileNotFoundError(f"Config file {self._config_file_name} not found")
        with open(self._config_file_name, "r") as file:
            return yaml.safe_load(file)

    def _validate_config(self, config):
        # Check for required top-level keys
        if "models_dir" not in config:
            raise KeyError("Missing 'models_dir' in the configuration")

        if "models" not in config:
            raise KeyError("Missing 'models' section in the configuration")

        models = config["models"]

        # Validate each model entry
        for model in models:
            if "name" not in model:
                raise KeyError("Missing 'name' in a model entry")
            if "subdir" not in model:
                raise KeyError("Missing 'subdir' in a model entry")

            # Validate types
            if not isinstance(model["name"], str):
                raise TypeError("'name' in a model entry must be a string")
            if not isinstance(model["subdir"], str):
                raise TypeError("'subdir' in a model entry must be a string")

        # Validate models_dir type
        if not isinstance(config["models_dir"], str):
            raise TypeError("'models_dir' must be a string")

    @property
    def models_dir(self):
        return self._models_dir

    @models_dir.setter
    def models_dir(self, value):
        if not isinstance(value, str):
            raise TypeError("models_dir must be a string")
        self._models_dir = value

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, value):
        if not isinstance(value, list):
            raise TypeError("models must be a list of model configurations")
        self._models = value
