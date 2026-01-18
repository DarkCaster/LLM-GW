"""
Unit tests for ModelSelector class.

Tests model selection logic, token estimation, and variant selection.
"""

import unittest
import asyncio
from unittest.mock import Mock, AsyncMock

from models.model_selector import ModelSelector


class TestModelSelector(unittest.TestCase):
    """Test cases for ModelSelector class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock configuration object
        self.mock_cfg = Mock()

        # Setup mock for model configuration
        # Mock get_table_seq to return model indices [1, 2, 3]
        self.mock_cfg.get_table_seq = Mock(return_value=[1, 2, 3])

        # Mock individual model configurations
        # Model 1: qwen3-30b-moe
        self.mock_cfg.get.side_effect = lambda key, default=None: {
            "models.1.name": "qwen3-30b-moe",
            "models.1.engine": "llama.cpp",
            "models.1.variants.1.binary": "/path/to/llama-server1",
            "models.1.variants.1.connect": "http://127.0.0.1:8080",
            "models.1.variants.1.tokenize": "true",
            "models.1.variants.1.context": "32000",
            "models.1.variants.2.binary": "/path/to/llama-server2",
            "models.1.variants.2.connect": "http://127.0.0.1:8081",
            "models.1.variants.2.tokenize": "true",
            "models.1.variants.2.context": "64000",
            "models.2.name": "mistral-7b",
            "models.2.engine": "llama.cpp",
            "models.2.variants.1.binary": "/path/to/llama-server3",
            "models.2.variants.1.connect": "http://127.0.0.1:8082",
            "models.2.variants.1.tokenize": "false",
            "models.2.variants.1.context": "8000",
            "models.3.name": "llama-70b",
            "models.3.engine": "llama.cpp",
            "models.3.variants.1.binary": "/path/to/llama-server4",
            "models.3.variants.1.connect": "http://127.0.0.1:8083",
            "models.3.variants.1.tokenize": "true",
            "models.3.variants.1.context": "16000",
        }.get(key, default)

        # Mock get_bool for tokenize
        self.mock_cfg.get_bool = Mock(
            side_effect=lambda key, default=None: {
                "models.1.variants.1.tokenize": True,
                "models.1.variants.2.tokenize": True,
                "models.2.variants.1.tokenize": False,
                "models.3.variants.1.tokenize": True,
            }.get(key, default)
        )

        # Mock get_int for context
        self.mock_cfg.get_int = Mock(
            side_effect=lambda key, default=None: {
                "models.1.variants.1.context": 32000,
                "models.1.variants.2.context": 64000,
                "models.2.variants.1.context": 8000,
                "models.3.variants.1.context": 16000,
            }.get(key, default)
        )

        # Mock get_list for args
        self.mock_cfg.get_list = Mock(return_value=["-c", "32000", "-m", "model.gguf"])

        # Mock variant sequence
        self.mock_cfg.get_table_seq.side_effect = lambda key: {
            "models.1.variants": [1, 2],
            "models.2.variants": [1],
            "models.3.variants": [1],
        }.get(key, [])

    def test_init_parses_models_correctly(self):
        """Test that ModelSelector correctly parses models from configuration."""
        selector = ModelSelector(self.mock_cfg)

        # Check that models are loaded
        self.assertEqual(len(selector._models), 3)

        # Check model names
        model_names = selector.list_models()
        self.assertIn("qwen3-30b-moe", model_names)
        self.assertIn("mistral-7b", model_names)
        self.assertIn("llama-70b", model_names)

        # Check variants are sorted by context
        qwen_variants = selector.get_all_variants("qwen3-30b-moe")
        self.assertEqual(len(qwen_variants), 2)
        self.assertEqual(qwen_variants[0]["context"], 32000)
        self.assertEqual(qwen_variants[1]["context"], 64000)

    def test_get_model_info(self):
        """Test getting information about a specific model."""
        selector = ModelSelector(self.mock_cfg)

        info = selector.get_model_info("qwen3-30b-moe")

        self.assertEqual(info["id"], "qwen3-30b-moe")
        self.assertEqual(info["engine"], "llama.cpp")
        self.assertEqual(info["variants_count"], 2)
        self.assertEqual(info["available_context_sizes"], [32000, 64000])
        self.assertTrue(info["supports_tokenization"])

    def test_get_model_info_not_found(self):
        """Test getting info for non-existent model raises error."""
        selector = ModelSelector(self.mock_cfg)

        with self.assertRaises(ValueError) as context:
            selector.get_model_info("non-existent-model")

        self.assertIn("not found in configuration", str(context.exception))

    def test_list_models(self):
        """Test listing all configured model names."""
        selector = ModelSelector(self.mock_cfg)

        models = selector.list_models()

        self.assertEqual(len(models), 3)
        self.assertIn("qwen3-30b-moe", models)
        self.assertIn("mistral-7b", models)
        self.assertIn("llama-70b", models)

    def test_get_model_engine_type(self):
        """Test getting engine type for a model."""
        selector = ModelSelector(self.mock_cfg)

        engine_type = selector.get_model_engine_type("qwen3-30b-moe")
        self.assertEqual(engine_type, "llama.cpp")

        with self.assertRaises(ValueError):
            selector.get_model_engine_type("non-existent")

    def test_get_available_models(self):
        """Test getting detailed information about all models."""
        selector = ModelSelector(self.mock_cfg)

        models_info = selector.get_available_models()

        self.assertEqual(len(models_info), 3)
        # Check all models are present
        model_names = {info["id"] for info in models_info}
        self.assertEqual(model_names, {"qwen3-30b-moe", "mistral-7b", "llama-70b"})

    async def test_select_variant_with_tokenization(self):
        """Test variant selection when tokenization is available."""
        selector = ModelSelector(self.mock_cfg)

        # Mock engine manager with current client
        mock_engine_manager = Mock()
        mock_client = AsyncMock()
        mock_client.estimate_tokens = AsyncMock(return_value=1000)

        mock_state = {
            "model_name": "qwen3-30b-moe",
            "variant_config": {"tokenize": True, "context": 32000},
        }
        mock_engine_manager.get_current_client = Mock(return_value=mock_client)
        mock_engine_manager.get_current_state = Mock(return_value=mock_state)

        # Test request data
        request_data = {
            "messages": [
                {"role": "user", "content": "Hello, how are you?" * 50}  # ~1000 chars
            ]
        }

        # Select variant
        variant = await selector.select_variant(
            "qwen3-30b-moe", request_data, mock_engine_manager
        )

        # Should select smallest variant with sufficient context
        # Estimated tokens: 1000 + safety margin (~512) = ~1512 < 32000
        self.assertEqual(variant["context"], 32000)
        mock_client.estimate_tokens.assert_called_once_with(request_data)

    async def test_select_variant_without_tokenization(self):
        """Test variant selection when tokenization is not available."""
        selector = ModelSelector(self.mock_cfg)

        # Mock engine manager without current client
        mock_engine_manager = Mock()
        mock_engine_manager.get_current_client = Mock(return_value=None)
        mock_engine_manager.get_current_state = Mock(return_value={})

        # Test request with chat messages
        request_data = {
            "messages": [
                {"role": "user", "content": "Hello" * 100}  # ~500 chars
            ],
            "max_tokens": 100,
        }

        # Select variant - should use fallback estimation
        variant = await selector.select_variant(
            "mistral-7b", request_data, mock_engine_manager
        )

        # Fallback: ~500 chars / 4 = 125 tokens + 100 max_tokens = 225
        # Safety margin: max(512, 22.5) = 512
        # Required: 225 + 512 = 737 < 8000
        self.assertEqual(variant["context"], 8000)

    async def test_select_variant_model_not_found(self):
        """Test selecting variant for non-existent model."""
        selector = ModelSelector(self.mock_cfg)
        mock_engine_manager = Mock()

        with self.assertRaises(ValueError) as context:
            await selector.select_variant(
                "non-existent", {"messages": []}, mock_engine_manager
            )

        self.assertIn("not found in configuration", str(context.exception))

    async def test_select_variant_insufficient_context(self):
        """Test selecting variant when no variant has sufficient context."""
        selector = ModelSelector(self.mock_cfg)

        # Mock engine manager
        mock_engine_manager = Mock()
        mock_client = AsyncMock()
        mock_client.estimate_tokens = AsyncMock(return_value=100000)  # Very large

        mock_state = {
            "model_name": "qwen3-30b-moe",
            "variant_config": {"tokenize": True, "context": 32000},
        }
        mock_engine_manager.get_current_client = Mock(return_value=mock_client)
        mock_engine_manager.get_current_state = Mock(return_value=mock_state)

        # Request requiring more tokens than any variant supports
        request_data = {
            "messages": [{"role": "user", "content": "Very long message" * 10000}]
        }

        with self.assertRaises(ValueError) as context:
            await selector.select_variant(
                "qwen3-30b-moe", request_data, mock_engine_manager
            )

        self.assertIn("Request requires", str(context.exception))
        self.assertIn("largest variant only supports", str(context.exception))

    async def test_select_variant_tokenization_fallback(self):
        """Test fallback when tokenization fails."""
        selector = ModelSelector(self.mock_cfg)

        # Mock engine manager with client that raises exception
        mock_engine_manager = Mock()
        mock_client = AsyncMock()
        mock_client.estimate_tokens = AsyncMock(
            side_effect=RuntimeError("Tokenization failed")
        )

        mock_state = {
            "model_name": "qwen3-30b-moe",
            "variant_config": {"tokenize": True, "context": 32000},
        }
        mock_engine_manager.get_current_client = Mock(return_value=mock_client)
        mock_engine_manager.get_current_state = Mock(return_value=mock_state)

        # Test request
        request_data = {
            "messages": [
                {"role": "user", "content": "Hello" * 10}  # ~50 chars
            ],
            "max_tokens": 50,
        }

        # Should use fallback estimation
        variant = await selector.select_variant(
            "qwen3-30b-moe", request_data, mock_engine_manager
        )

        # Should still select a variant
        self.assertIn(variant["context"], [32000, 64000])
        mock_client.estimate_tokens.assert_called_once()

    def test_fallback_token_estimation_chat(self):
        """Test fallback token estimation for chat completions."""
        selector = ModelSelector(self.mock_cfg)

        # Chat completions request
        request_data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
            "max_tokens": 100,
        }

        tokens = selector._fallback_token_estimation(request_data)

        # Total characters: 5 + 9 + 12 = 26
        # Prompt tokens: ceil(26 / 4) = 7
        # Total: 7 + 100 = 107
        self.assertEqual(tokens, 107)

    def test_fallback_token_estimation_text(self):
        """Test fallback token estimation for text completions."""
        selector = ModelSelector(self.mock_cfg)

        # Text completions request
        request_data = {
            "prompt": "Once upon a time, there was a brave knight.",
            "max_tokens": 50,
        }

        tokens = selector._fallback_token_estimation(request_data)

        # Characters: ~43
        # Prompt tokens: ceil(43 / 4) = 11
        # Total: 11 + 50 = 61
        self.assertGreaterEqual(tokens, 61)
        self.assertLessEqual(
            tokens, 65
        )  # Allow for slight variation in character count

    def test_fallback_token_estimation_prompt_list(self):
        """Test fallback token estimation with prompt list."""
        selector = ModelSelector(self.mock_cfg)

        request_data = {
            "prompt": ["First part.", "Second part is longer."],
            "max_tokens": 30,
        }

        tokens = selector._fallback_token_estimation(request_data)

        # Should handle list prompts correctly
        self.assertGreater(tokens, 30)

    def test_fallback_token_estimation_minimum(self):
        """Test fallback token estimation returns minimum tokens."""
        selector = ModelSelector(self.mock_cfg)

        # Empty request
        request_data = {}
        tokens = selector._fallback_token_estimation(request_data)

        # Should return minimum of 10 tokens
        self.assertEqual(tokens, 10)


# Async test runner
if __name__ == "__main__":
    # Run async tests
    async def run_async_tests():
        """Run async test methods."""
        test_case = TestModelSelector()

        # Run async tests
        await test_case.test_select_variant_with_tokenization()
        await test_case.test_select_variant_without_tokenization()
        await test_case.test_select_variant_model_not_found()
        await test_case.test_select_variant_insufficient_context()
        await test_case.test_select_variant_tokenization_fallback()

        print("All async tests passed!")

    # Run sync tests
    unittest.main(verbosity=2)

    # Run async tests
    asyncio.run(run_async_tests())
