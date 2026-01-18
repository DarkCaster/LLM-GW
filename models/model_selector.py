"""
ModelSelector - Selects appropriate model variant based on request requirements.

Analyzes incoming requests, estimates token requirements, and selects the
smallest sufficient variant from available model configurations.
"""

import logging
import math
import python_lua_helper
from typing import Dict, List, Any

from engine.engine_manager import EngineManager


class ModelSelector:
    """Selects model variants based on context requirements and request analysis."""

    def __init__(self, cfg: python_lua_helper.PyLuaHelper):
        """
        Initialize ModelSelector with configuration.

        Args:
            cfg: PyLuaHelper configuration object
        """
        self._cfg = cfg
        self._logger = logging.getLogger(self.__class__.__name__)

        # Parse and index all models from configuration
        self._models = self._parse_models()

        self._logger.info(f"ModelSelector initialized with {len(self._models)} models")

    def _parse_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Parse models from configuration and build internal data structure.

        Returns:
            Dictionary mapping model names to model data with variants
        """
        models_dict = {}

        # Get list of model indices from configuration
        model_indices = self._cfg.get_table_seq("models")

        for idx in model_indices:
            model_key = f"models.{idx}"

            try:
                # Get model name (mandatory field)
                model_name = self._cfg.get(f"{model_key}.name")
                if not model_name:
                    self._logger.warning(f"Model at index {idx} has no name, skipping")
                    continue

                # Get engine type (mandatory field)
                engine_type = self._cfg.get(f"{model_key}.engine")
                if not engine_type:
                    self._logger.warning(
                        f"Model '{model_name}' has no engine type, skipping"
                    )
                    continue

                # Get variants
                variants = self._parse_variants(model_key, model_name)
                if not variants:
                    self._logger.warning(
                        f"Model '{model_name}' has no valid variants, skipping"
                    )
                    continue

                # Sort variants by context size (smallest to largest)
                sorted_variants = sorted(variants, key=lambda v: v.get("context", 0))

                models_dict[model_name] = {
                    "name": model_name,
                    "engine": engine_type,
                    "variants": sorted_variants,
                }

                self._logger.debug(
                    f"Loaded model '{model_name}' with {len(sorted_variants)} variants, "
                    f"context sizes: {[v.get('context', 0) for v in sorted_variants]}"
                )

            except Exception as e:
                self._logger.error(f"Failed to parse model at index {idx}: {e}")
                continue

        return models_dict

    def _parse_variants(self, model_key: str, model_name: str) -> List[Dict[str, Any]]:
        """
        Parse variants for a specific model.

        Args:
            model_key: Configuration key for the model
            model_name: Name of the model (for logging)

        Returns:
            List of variant configurations
        """
        variants = []

        try:
            # Get variant indices for this model
            variant_indices = self._cfg.get_table_seq(f"{model_key}.variants")

            for var_idx in variant_indices:
                variant_key = f"{model_key}.variants.{var_idx}"

                try:
                    # Extract variant configuration
                    variant_config = {
                        "binary": self._cfg.get(f"{variant_key}.binary"),
                        "connect": self._cfg.get(f"{variant_key}.connect"),
                        "args": self._cfg.get_list(f"{variant_key}.args"),
                        "tokenize": self._cfg.get_bool(
                            f"{variant_key}.tokenize", False
                        ),
                        "context": self._cfg.get_int(f"{variant_key}.context", 0),
                    }

                    # Validate required fields
                    if not variant_config["binary"] or not variant_config["connect"]:
                        self._logger.warning(
                            f"Variant {var_idx} of model '{model_name}' missing required fields, skipping"
                        )
                        continue

                    variants.append(variant_config)

                except Exception as e:
                    self._logger.error(
                        f"Failed to parse variant {var_idx} for model '{model_name}': {e}"
                    )
                    continue

        except Exception as e:
            self._logger.error(
                f"Failed to parse variants for model '{model_name}': {e}"
            )

        return variants

    async def select_variant(
        self,
        model_name: str,
        request_data: Dict[str, Any],
        engine_manager: EngineManager,
    ) -> Dict[str, Any]:
        """
        Select appropriate model variant based on request requirements.

        Args:
            model_name: Name of the model to use
            request_data: Request data dictionary (from OpenAI API)
            engine_manager: EngineManager instance for token estimation

        Returns:
            Selected variant configuration

        Raises:
            ValueError: If model not found or no variant has sufficient context
        """
        # Validate model exists
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not found in configuration")

        model_info = self._models[model_name]
        variants = model_info["variants"]

        self._logger.debug(
            f"Selecting variant for model '{model_name}', "
            f"available contexts: {[v.get('context', 0) for v in variants]}"
        )

        # Estimate required context size
        estimated_tokens = await self._estimate_required_tokens(
            model_name, request_data, engine_manager
        )

        # Add safety margin (10% + 512 tokens minimum)
        safety_margin = max(512, int(estimated_tokens * 0.1))
        required_tokens = estimated_tokens + safety_margin

        self._logger.debug(
            f"Token estimation: {estimated_tokens} + {safety_margin} safety = {required_tokens} required"
        )

        # Select smallest variant with sufficient context
        selected_variant = None
        for variant in variants:
            context_size = variant.get("context", 0)
            if context_size >= required_tokens:
                selected_variant = variant
                break

        if not selected_variant:
            max_context = max(v.get("context", 0) for v in variants)
            raise ValueError(
                f"Request requires {required_tokens} tokens, "
                f"but largest variant only supports {max_context}"
            )

        self._logger.info(
            f"Selected variant for model '{model_name}': "
            f"context={selected_variant.get('context', 0)}, "
            f"estimated={estimated_tokens}, required={required_tokens}"
        )

        return selected_variant

    async def _estimate_required_tokens(
        self,
        model_name: str,
        request_data: Dict[str, Any],
        engine_manager: EngineManager,
    ) -> int:
        """
        Estimate token requirements for the request.

        Args:
            model_name: Name of the model
            request_data: Request data dictionary
            engine_manager: EngineManager instance

        Returns:
            Estimated token count
        """
        model_info = self._models[model_name]
        current_client = engine_manager.get_current_client()

        # Check if we can use precise tokenization
        can_tokenize = False
        current_variant_config = None

        # Get current engine state
        current_state = engine_manager.get_current_state()
        if current_state.get("model_name") == model_name and current_state.get(
            "variant_config"
        ):
            current_variant_config = current_state["variant_config"]

        # Check if current variant supports tokenization
        if (
            current_client
            and current_variant_config
            and current_variant_config.get("tokenize", False)
        ):
            can_tokenize = True

        if can_tokenize:
            try:
                # Use engine's precise token estimation
                tokens = await current_client.estimate_tokens(request_data)
                self._logger.debug(f"Precise token estimation: {tokens} tokens")
                return tokens
            except Exception as e:
                self._logger.warning(
                    f"Precise token estimation failed: {e}, using fallback"
                )
                # Fall through to fallback estimation

        # Fallback: rough estimation based on character count
        # Assume ~4 characters per token (conservative estimate for English)
        fallback_tokens = self._fallback_token_estimation(request_data)

        self._logger.warning(
            f"Using fallback token estimation: {fallback_tokens} tokens "
            f"(model: {model_name})"
        )

        return fallback_tokens

    def _fallback_token_estimation(self, request_data: Dict[str, Any]) -> int:
        """
        Fallback token estimation based on character count.

        Args:
            request_data: Request data dictionary

        Returns:
            Rough token estimate
        """
        total_chars = 0

        # Check for chat completions request
        if "messages" in request_data:
            messages = request_data.get("messages", [])
            for message in messages:
                content = message.get("content", "")
                if content:
                    total_chars += len(str(content))

        # Check for text completions request
        elif "prompt" in request_data:
            prompt = request_data.get("prompt", "")
            if isinstance(prompt, list):
                for p in prompt:
                    total_chars += len(str(p))
            else:
                total_chars += len(str(prompt))

        # Add max_tokens if specified
        max_tokens = request_data.get("max_tokens", 0)

        # Estimate tokens: characters / 4 (rough estimate for English)
        prompt_tokens = math.ceil(total_chars / 4) if total_chars > 0 else 0
        total_tokens = prompt_tokens + max_tokens

        # Ensure minimum token count
        return max(total_tokens, 10)

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model information

        Raises:
            ValueError: If model not found
        """
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not found in configuration")

        model_info = self._models[model_name]

        return {
            "id": model_name,
            "object": "model",
            "created": 0,  # Placeholder
            "owned_by": "llm-gateway",
            "engine": model_info["engine"],
            "available_context_sizes": [
                v.get("context", 0) for v in model_info["variants"]
            ],
            "variants_count": len(model_info["variants"]),
            "supports_tokenization": any(
                v.get("tokenize", False) for v in model_info["variants"]
            ),
        }

    def list_models(self) -> List[str]:
        """
        Get list of all configured model names.

        Returns:
            List of model names
        """
        return list(self._models.keys())

    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get detailed information about all configured models.

        Returns:
            List of model information dictionaries
        """
        models_info = []
        for model_name in self.list_models():
            try:
                model_info = self.get_model_info(model_name)
                models_info.append(model_info)
            except Exception as e:
                self._logger.error(f"Failed to get info for model '{model_name}': {e}")

        return models_info

    def get_model_engine_type(self, model_name: str) -> str:
        """
        Get engine type for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Engine type string

        Raises:
            ValueError: If model not found
        """
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not found in configuration")

        return self._models[model_name]["engine"]

    def get_all_variants(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get all variants for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            List of variant configurations

        Raises:
            ValueError: If model not found
        """
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not found in configuration")

        return self._models[model_name]["variants"].copy()
