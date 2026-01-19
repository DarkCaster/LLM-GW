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
        self.logger.info(f"Selecting variant for model '{model_name}'")

        # Find model in configuration
        model_index = None
        for i in self.cfg.get_table_seq("models"):
            if self.cfg.get(f"models.{i}.name") == model_name:
                model_index = i
                break

        if model_index is None:
            raise ValueError(f"Model '{model_name}' not found in configuration")

        # Get engine type for this model
        engine_type = self.cfg.get(f"models.{model_index}.engine")

        # Only support llama.cpp for now
        if engine_type != "llama.cpp":
            raise ValueError(
                f"Engine type '{engine_type}' not supported yet. "
                "Only 'llama.cpp' is currently supported."
            )

        self.logger.info(
            f"Found model '{model_name}' at index {model_index} with engine type '{engine_type}'"
        )

        # Construct required_config for context estimation
        required_config = {
            "operation": "context_estimation",
        }

        # Get engine client for context size estimation
        self.logger.info("Getting engine client for context estimation")
        estimation_client = await self.engine_manager.ensure_engine(
            model_name, required_config, engine_type
        )

        # Estimate tokens
        self.logger.info("Estimating token requirements")
        context_size_required = await estimation_client.estimate_tokens(request_data)
        self.logger.info(f"Context size required: {context_size_required} tokens")

        # Iterate over model's variants and select first suitable one
        variant_index = None
        for i in self.cfg.get_table_seq(f"models.{model_index}.variants"):
            variant_context = self.cfg.get_int(
                f"models.{model_index}.variants.{i}.context", 0
            )
            self.logger.debug(
                f"Checking variant {i}: context={variant_context}, required={context_size_required}"
            )

            if variant_context >= context_size_required:
                variant_index = i
                self.logger.info(
                    f"Selected variant {variant_index} with context size {variant_context}"
                )
                break

        if variant_index is None:
            raise ValueError(
                f"No suitable variant found for model '{model_name}' "
                f"with required context size {context_size_required}"
            )

        # Construct required_config for final operation
        required_config = {
            "operation": "text_query",
            "variant_index": variant_index,
            "context_size_required": context_size_required,
        }

        # Get engine client for final operation
        self.logger.info(
            f"Getting engine client for text query with variant {variant_index}"
        )
        final_client = await self.engine_manager.ensure_engine(
            model_name, required_config, engine_type
        )

        if final_client is None:
            raise ValueError(
                f"Failed to get engine client for model '{model_name}' variant {variant_index}"
            )

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
