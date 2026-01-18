# models/model_selector.py

from typing import Dict, List
from utils.logger import get_logger
from engine import EngineManager


class ModelSelector:
    """
    Analyze requests and select appropriate model variant based on context requirements.
    """

    def __init__(self, cfg):
        """
        Initialize the model selector.

        Args:
            cfg: PyLuaHelper configuration object
        """
        self.cfg = cfg
        self.logger = get_logger(self.__class__.__name__)

        # Internal data structure: {model_name: {engine: str, variants: [variant_configs]}}
        self.models: Dict[str, Dict] = {}

        # Parse and index all models from configuration
        self._parse_models()

    def _parse_models(self) -> None:
        """
        Parse and index all models from configuration.
        Build internal data structure with models and their variants.
        """
        self.logger.info("Parsing models from configuration")

        # Get the number of models in the models table
        models_count = self.cfg.get_table_end("models") - self.cfg.get_table_start(
            "models"
        )

        if models_count == 0:
            self.logger.warning("No models found in configuration")
            return

        # Iterate through models table
        for i in self.cfg.get_table_seq("models"):
            model_path = f"models.{i}"

            # Extract model name
            model_name = self.cfg.get(f"{model_path}.name")
            if not model_name:
                self.logger.warning(f"Model at index {i} has no name, skipping")
                continue

            # Extract engine type
            engine_type = self.cfg.get(f"{model_path}.engine")
            if not engine_type:
                self.logger.warning(
                    f"Model '{model_name}' has no engine type, skipping"
                )
                continue

            # Parse variants
            variants = []
            variants_path = f"{model_path}.variants"

            for variant_idx in self.cfg.get_table_seq(variants_path):
                variant_path = f"{variants_path}.{variant_idx}"

                # Extract variant configuration
                variant_config = {
                    "binary": self.cfg.get(f"{variant_path}.binary"),
                    "connect": self.cfg.get(f"{variant_path}.connect"),
                    "args": self.cfg.get_list(f"{variant_path}.args"),
                    "tokenize": self.cfg.get_bool(f"{variant_path}.tokenize", False),
                    "context": self.cfg.get_int(f"{variant_path}.context", 0),
                }

                variants.append(variant_config)

            # Sort variants by context size (smallest to largest) for efficient selection
            variants.sort(key=lambda v: v["context"])

            # Store model information
            self.models[model_name] = {"engine": engine_type, "variants": variants}

            self.logger.info(
                f"Loaded model '{model_name}' with engine '{engine_type}' "
                f"and {len(variants)} variant(s)"
            )

        self.logger.info(f"Parsed {len(self.models)} model(s) from configuration")

    async def select_variant(
        self, model_name: str, request_data: dict, engine_manager: EngineManager
    ) -> dict:
        """
        Select appropriate model variant based on request requirements.

        Args:
            model_name: Name of the requested model
            request_data: The incoming request data dictionary
            engine_manager: EngineManager instance for accessing current engine

        Returns:
            Selected variant configuration dictionary

        Raises:
            ValueError: If model not found or no suitable variant available
        """
        # Validate that model exists
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in configuration")

        model_info = self.models[model_name]
        variants = model_info["variants"]

        if not variants:
            raise ValueError(f"Model '{model_name}' has no configured variants")

        # Estimate required context size
        estimated_tokens = await self._estimate_context_size(
            model_name, request_data, engine_manager
        )

        # Add safety margin (10% or minimum 512 tokens, whichever is larger)
        safety_margin = max(int(estimated_tokens * 0.1), 512)
        required_tokens = estimated_tokens + safety_margin

        self.logger.info(
            f"Estimated tokens: {estimated_tokens}, "
            f"with safety margin: {required_tokens}"
        )

        # Select smallest variant where variant.context >= required_tokens
        selected_variant = None

        for variant in variants:
            variant_context = variant["context"]
            if variant_context >= required_tokens:
                selected_variant = variant
                self.logger.info(
                    f"Selected variant with context size {variant_context} "
                    f"for model '{model_name}' (required: {required_tokens})"
                )
                break

        # If no variant found, raise error
        if selected_variant is None:
            max_context = variants[-1]["context"] if variants else 0
            raise ValueError(
                f"Request requires {required_tokens} tokens, but largest variant "
                f"for model '{model_name}' only supports {max_context} tokens"
            )

        return selected_variant

    async def _estimate_context_size(
        self, model_name: str, request_data: dict, engine_manager: EngineManager
    ) -> int:
        """
        Estimate the context size required for the request.

        Args:
            model_name: Name of the requested model
            request_data: The incoming request data dictionary
            engine_manager: EngineManager instance for accessing current engine

        Returns:
            Estimated number of tokens required
        """
        # Check if there's a currently running engine that supports tokenization
        current_client = engine_manager.get_current_client()

        # Check if current engine is for the same model and supports tokenization
        can_use_tokenization = False

        if current_client is not None:
            # Check if current engine is for the same model
            if engine_manager.current_model_name == model_name:
                # Check if the variant supports tokenization
                if engine_manager.current_variant_config:
                    can_use_tokenization = engine_manager.current_variant_config.get(
                        "tokenize", False
                    )

        if can_use_tokenization:
            # Use engine's tokenization endpoint
            try:
                self.logger.debug("Using engine tokenization for estimation")
                estimated_tokens = await current_client.estimate_tokens(request_data)
                return estimated_tokens
            except Exception as e:
                self.logger.warning(
                    f"Failed to use engine tokenization: {e}, falling back to estimation"
                )

        # Fallback to rough estimation
        self.logger.debug("Using fallback character-based estimation")
        return self._fallback_estimate(request_data)

    def _fallback_estimate(self, request_data: dict) -> int:
        """
        Fallback method to estimate tokens based on character count.
        Uses rough heuristic: 1 token ≈ 4 characters.

        Args:
            request_data: The incoming request data dictionary

        Returns:
            Estimated number of tokens
        """
        char_count = 0

        # Determine request type
        if "messages" in request_data:
            # Chat completions
            messages = request_data.get("messages", [])
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    char_count += len(content)
                elif isinstance(content, list):
                    # Handle array of content parts
                    for part in content:
                        if isinstance(part, dict):
                            text = part.get("text", "")
                            char_count += len(text)
        elif "prompt" in request_data:
            # Text completions
            prompt = request_data.get("prompt", "")
            if isinstance(prompt, str):
                char_count += len(prompt)
            elif isinstance(prompt, list):
                for p in prompt:
                    char_count += len(str(p))

        # Rough estimation: 1 token ≈ 4 characters
        estimated_prompt_tokens = char_count // 4

        # Add max_tokens if present
        max_tokens = request_data.get("max_tokens", 0)

        total_estimate = estimated_prompt_tokens + max_tokens

        self.logger.warning(
            f"Using imprecise estimation: {char_count} chars ≈ "
            f"{estimated_prompt_tokens} tokens + {max_tokens} max_tokens = "
            f"{total_estimate} total"
        )

        return total_estimate

    def get_model_info(self, model_name: str) -> dict:
        """
        Get information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model information

        Raises:
            ValueError: If model not found
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in configuration")

        model_info = self.models[model_name]
        variants = model_info["variants"]

        # Extract context sizes from variants
        context_sizes = [v["context"] for v in variants]

        return {
            "name": model_name,
            "engine": model_info["engine"],
            "variants_count": len(variants),
            "context_sizes": context_sizes,
            "min_context": min(context_sizes) if context_sizes else 0,
            "max_context": max(context_sizes) if context_sizes else 0,
        }

    def list_models(self) -> List[str]:
        """
        Get list of all configured model names.

        Returns:
            List of model name strings
        """
        return list(self.models.keys())
