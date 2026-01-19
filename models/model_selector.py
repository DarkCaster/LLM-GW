# models/model_selector.py

import python_lua_helper
from utils.logger import get_logger
from typing import List
from engine import EngineManager, EngineClient


class ModelSelector:
    """
    Analyze requests and select appropriate model variant based on context requirements.
    """

    def __init__(
        self, engine_manager: EngineManager, cfg: python_lua_helper.PyLuaHelper
    ):
        """
        Initialize ModelSelector.

        Args:
            engine_manager: EngineManager instance for managing engine lifecycle
            cfg: PyLuaHelper configuration object
        """
        self.engine_manager = engine_manager
        self.cfg = cfg
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("ModelSelector initialized")

    # TODO: different operations support depending on request_data (for now its always text query for generating chat completions)
    async def select_variant(self, model_name: str, request_data: dict) -> EngineClient:
        """
        Select appropriate model variant based on context requirements.

        Args:
            model_name: Name of the model to select
            request_data: Dictionary containing the request data from await request.json()

        Returns:
            EngineClient instance for the selected variant

        Raises:
            ValueError: If model not found, engine type not supported, or variant selection fails
        """
        self.logger.info(f"Selecting suitable configuration for model '{model_name}'")

        # Construct required_config for context estimation
        required_config = {
            "operation": "context_estimation",
            "context_size_required": 0,
        }

        # Get engine client for context size estimation
        self.logger.info("Getting engine client for context estimation")
        estimation_client = await self.engine_manager.ensure_engine(
            model_name, required_config
        )

        # Estimate tokens
        self.logger.info("Estimating token requirements")
        context_size_required = await estimation_client.estimate_tokens(request_data)
        self.logger.info(f"Context size required: {context_size_required} tokens")

        # Construct required_config for final operation
        required_config = {
            "operation": "text_query",
            "context_size_required": context_size_required,
        }

        # Get engine client for final operation
        self.logger.info("Getting engine client for text query")
        final_client = await self.engine_manager.ensure_engine(
            model_name, required_config
        )

        if final_client is None:
            raise ValueError(f"Failed to get engine client for model '{model_name}'")

        variant_index = required_config.get("variant_index", 0)
        self.logger.info(
            f"Successfully selected and loaded variant {variant_index} for model '{model_name}'"
        )
        return final_client

    def list_models(self) -> List[str]:
        """
        Return list of all configured model names.

        Returns:
            List of model names from configuration
        """
        model_names = []

        for i in self.cfg.get_table_seq("models"):
            model_name = self.cfg.get(f"models.{i}.name")
            if model_name:
                model_names.append(model_name)

        self.logger.debug(f"Listed {len(model_names)} models: {model_names}")
        return model_names
