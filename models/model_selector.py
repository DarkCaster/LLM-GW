# models/model_selector.py

import python_lua_helper
import logger
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
        self.logger = logger.get_logger(self.__class__.__name__)
        self.logger.info("ModelSelector initialized")

    async def select_variant(
        self, path: str, model_name: str, request_data: dict
    ) -> tuple[EngineClient, float]:
        """
        Select appropriate model and variant based on model name, request type, context requirements and other properties.

        Args:
            model_name: Name of the model to select
            request_data: Dictionary containing the request data from await request.json()

        Returns:
            EngineClient instance for the selected variant with proposed idle timeout

        Raises:
            ValueError: If model not found, engine type not supported, or variant selection fails
        """
        self.logger.debug(f"Selecting suitable configuration for model '{model_name}'")

        # TODO: add more path-specific logic if needed
        if path == "/v1/embeddings":
            # For embedding requests just load first available configuration and use it
            config = {"operation": "text_query", "context_size_required": 0}
            # Get engine client for final operation
            client, timeout = await self.engine_manager.ensure_engine(
                model_name, config
            )
            if client is None:
                raise ValueError(
                    f"Failed to get engine client for model '{model_name}'"
                )
            # Return engine-client to use engine we just started
            return client, timeout
        else:
            # For all other requests, typical pipeline is:
            # - tokenize request-contents to estimate context size requirements
            # - select suitable model-variant configuration and start engine for it

            # Try getting standalone tokenizer for quick estimation without starting the model
            standalone_tokenizer = await self.engine_manager.ensure_local_tokenizer(
                model_name
            )
            context_size_required = 0
            # Tokenize with fast standalone tokenizer if available
            if standalone_tokenizer is not None:
                self.logger.debug(
                    "Estimating token requirements with standalone tokenizer"
                )
                context_size_required = await standalone_tokenizer.estimate_tokens(
                    request_data
                )
                self.logger.info(
                    f"Context size required (standalone tokenizer): {context_size_required} tokens"
                )

            # Construct required_config for context estimation
            estimation_config = {
                "operation": "text_query",
                "context_size_required": context_size_required,
            }
            # Get engine client for context size estimation
            estimation_client, _ = await self.engine_manager.ensure_engine(
                model_name, estimation_config
            )
            # Estimate tokens
            context_size_required = await estimation_client.estimate_tokens(
                request_data
            )
            self.logger.info(f"Context size required: {context_size_required} tokens")

            # Construct required_config for operation we wanted
            text_query_config = {
                "operation": "text_query",
                "context_size_required": context_size_required,
            }
            # Get engine client for final operation
            final_client, idle_timeout = await self.engine_manager.ensure_engine(
                model_name, text_query_config
            )
            if final_client is None:
                raise ValueError(
                    f"Failed to get engine client for model '{model_name}'"
                )

            # Return engine-client to use engine we just started
            return final_client, idle_timeout

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
