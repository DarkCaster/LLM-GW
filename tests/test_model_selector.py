import unittest
from unittest.mock import AsyncMock, MagicMock

import python_lua_helper

from models.model_selector import ModelSelector
from engine.engine_manager import EngineManager


class TestModelSelector(unittest.IsolatedAsyncioTestCase):
    """Test the ModelSelector class for model variant selection."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock PyLuaHelper configuration
        self.mock_cfg = MagicMock(spec=python_lua_helper.PyLuaHelper)

        # Set up mock configuration structure
        self._setup_mock_configuration()

        # Create ModelSelector with mock config
        self.selector = ModelSelector(self.mock_cfg)

        # Test data
        self.model_name = "qwen3-30b-moe"
        self.engine_type = "llama.cpp"

        # Mock EngineManager
        self.mock_engine_manager = MagicMock(spec=EngineManager)

    def _setup_mock_configuration(self):
        """Set up the mock Lua configuration structure."""
        # Mock model indices
        self.mock_cfg.get_table_seq = MagicMock()

        # For models table, return [1] (one model at index 1)
        def get_table_seq_side_effect(key):
            if key == "models":
                return [1]
            elif key == "models.1.variants":
                return [1, 2]  # Two variants
            elif key == "models.1.variants.1.args":
                return []
            elif key == "models.1.variants.2.args":
                return []
            return []

        self.mock_cfg.get_table_seq.side_effect = get_table_seq_side_effect

        # Mock get methods for model 1
        def get_side_effect(key, default=None):
            if key == "models.1.name":
                return "qwen3-30b-moe"
            elif key == "models.1.engine":
                return "llama.cpp"
            elif key == "models.1.variants.1.binary":
                return "/path/to/llama-server"
            elif key == "models.1.variants.1.connect":
                return "http://127.0.0.1:8080"
            elif key == "models.1.variants.1.tokenize":
                return "true"
            elif key == "models.1.variants.1.context":
                return "32000"
            elif key == "models.1.variants.2.binary":
                return "/path/to/llama-server"
            elif key == "models.1.variants.2.connect":
                return "http://127.0.0.1:8080"
            elif key == "models.1.variants.2.tokenize":
                return "true"
            elif key == "models.1.variants.2.context":
                return "64000"
            return default

        self.mock_cfg.get.side_effect = get_side_effect

        # Mock get_bool for tokenize
        def get_bool_side_effect(key, default=False):
            if "tokenize" in key:
                return True
            return default

        self.mock_cfg.get_bool.side_effect = get_bool_side_effect

        # Mock get_int for context
        def get_int_side_effect(key, default=0):
            if key == "models.1.variants.1.context":
                return 32000
            elif key == "models.1.variants.2.context":
                return 64000
            return default

        self.mock_cfg.get_int.side_effect = get_int_side_effect

        # Mock get_list for args
        def get_list_side_effect(key):
            if key == "models.1.variants.1.args":
                return ["-c", "32000", "-m", "model.gguf"]
            elif key == "models.1.variants.2.args":
                return ["-c", "64000", "-m", "model.gguf"]
            return []

        self.mock_cfg.get_list.side_effect = get_list_side_effect

    def test_model_parsing_from_configuration(self):
        """Test that models are correctly parsed from configuration."""
        # Verify model was loaded
        self.assertIn(self.model_name, self.selector._models)

        model_info = self.selector._models[self.model_name]

        # Check model information
        self.assertEqual(model_info["name"], self.model_name)
        self.assertEqual(model_info["engine"], self.engine_type)

        # Check variants were loaded and sorted
        variants = model_info["variants"]
        self.assertEqual(len(variants), 2)

        # Check variant contexts (should be sorted)
        self.assertEqual(variants[0]["context"], 32000)
        self.assertEqual(variants[1]["context"], 64000)

        # Check variant configurations
        variant1 = variants[0]
        self.assertEqual(variant1["binary"], "/path/to/llama-server")
        self.assertEqual(variant1["connect"], "http://127.0.0.1:8080")
        self.assertEqual(variant1["tokenize"], True)
        self.assertEqual(variant1["context"], 32000)
        self.assertEqual(variant1["args"], ["-c", "32000", "-m", "model.gguf"])

    def test_parse_models_empty_configuration(self):
        """Test parsing when configuration has no models."""
        mock_cfg = MagicMock(spec=python_lua_helper.PyLuaHelper)
        mock_cfg.get_table_seq.return_value = []  # No models

        selector = ModelSelector(mock_cfg)

        # Should have no models loaded
        self.assertEqual(len(selector._models), 0)
        self.assertEqual(selector.list_models(), [])

    def test_parse_models_missing_required_fields(self):
        """Test parsing when model is missing required fields."""
        mock_cfg = MagicMock(spec=python_lua_helper.PyLuaHelper)

        # Return one model index
        mock_cfg.get_table_seq.return_value = [1]

        # Model missing name
        mock_cfg.get.side_effect = lambda key, default=None: default

        selector = ModelSelector(mock_cfg)

        # Model should be skipped
        self.assertEqual(len(selector._models), 0)

    def test_parse_variants_empty(self):
        """Test parsing when model has no variants."""
        mock_cfg = MagicMock(spec=python_lua_helper.PyLuaHelper)

        # Return one model index
        mock_cfg.get_table_seq.side_effect = lambda key: [1] if key == "models" else []

        # Model with name and engine but no variants
        mock_cfg.get.side_effect = lambda key, default=None: (
            "test-model"
            if key == "models.1.name"
            else "llama.cpp"
            if key == "models.1.engine"
            else default
        )

        selector = ModelSelector(mock_cfg)

        # Model should be skipped due to no variants
        self.assertEqual(len(selector._models), 0)

    async def test_select_variant_sufficient_context(self):
        """Test variant selection when variant has sufficient context."""
        # Mock engine manager state
        mock_state = {
            "model_name": self.model_name,
            "variant_config": {
                "tokenize": True,
                "context": 32000,
            },
            "engine_running": True,
        }

        mock_client = AsyncMock()
        mock_client.estimate_tokens = AsyncMock(return_value=1000)

        self.mock_engine_manager.get_current_state.return_value = mock_state
        self.mock_engine_manager.get_current_client.return_value = mock_client

        # Request data requiring 1000 tokens
        request_data = {
            "messages": [
                {"role": "user", "content": "Hello, how are you?" * 10},  # ~200 chars
                {
                    "role": "assistant",
                    "content": "I'm fine, thank you!" * 10,
                },  # ~200 chars
            ],
            "max_tokens": 500,
        }

        # Select variant
        variant = await self.selector.select_variant(
            model_name=self.model_name,
            request_data=request_data,
            engine_manager=self.mock_engine_manager,
        )

        # Should select smallest variant with sufficient context (32000)
        # Estimated tokens: ~400 chars / 4 = 100 tokens + 500 max_tokens = 600
        # Safety margin: max(512, 60) = 512
        # Required: 600 + 512 = 1112 < 32000
        self.assertEqual(variant["context"], 32000)

        # Token estimation should have been called
        mock_client.estimate_tokens.assert_called_once_with(request_data)

    async def test_select_variant_requires_larger_context(self):
        """Test variant selection when larger context is needed."""
        # Mock engine manager state
        mock_state = {
            "model_name": self.model_name,
            "variant_config": {
                "tokenize": True,
                "context": 32000,
            },
            "engine_running": True,
        }

        mock_client = AsyncMock()
        # Simulate large token requirement (50,000 tokens)
        mock_client.estimate_tokens = AsyncMock(return_value=50000)

        self.mock_engine_manager.get_current_state.return_value = mock_state
        self.mock_engine_manager.get_current_client.return_value = mock_client

        # Request data requiring many tokens
        request_data = {
            "messages": [
                {"role": "user", "content": "A" * 10000},  # Large content
            ],
            "max_tokens": 1000,
        }

        # Select variant
        variant = await self.selector.select_variant(
            model_name=self.model_name,
            request_data=request_data,
            engine_manager=self.mock_engine_manager,
        )

        # Should select larger variant (64000)
        # Estimated tokens: 50000 + 1000 = 51000
        # Safety margin: max(512, 5100) = 5100
        # Required: 51000 + 5100 = 56100 < 64000
        self.assertEqual(variant["context"], 64000)

    async def test_select_variant_insufficient_context(self):
        """Test variant selection when no variant has sufficient context."""
        # Create selector with only small variant (32000 context)
        # Modify mock to only return one small variant
        self.mock_cfg.get_table_seq.side_effect = (
            lambda key: [1] if key == "models.1.variants" else [1]
        )
        self.mock_cfg.get_int.side_effect = (
            lambda key, default=0: 32000 if "context" in key else default
        )

        selector = ModelSelector(self.mock_cfg)

        # Mock engine manager state
        mock_state = {
            "model_name": self.model_name,
            "variant_config": {
                "tokenize": True,
                "context": 32000,
            },
            "engine_running": True,
        }

        mock_client = AsyncMock()
        # Simulate huge token requirement (100,000 tokens)
        mock_client.estimate_tokens = AsyncMock(return_value=100000)

        self.mock_engine_manager.get_current_state.return_value = mock_state
        self.mock_engine_manager.get_current_client.return_value = mock_client

        # Request data requiring more tokens than available
        request_data = {
            "messages": [
                {"role": "user", "content": "A" * 50000},
            ],
            "max_tokens": 10000,
        }

        # Should raise ValueError
        with self.assertRaises(ValueError) as cm:
            await selector.select_variant(
                model_name=self.model_name,
                request_data=request_data,
                engine_manager=self.mock_engine_manager,
            )

        self.assertIn("largest variant only supports", str(cm.exception))

    async def test_select_variant_fallback_estimation(self):
        """Test fallback token estimation when tokenization is unavailable."""
        # Mock engine manager with no current client (can't tokenize)
        mock_state = {
            "model_name": "different-model",  # Different model, so tokenization not available
            "variant_config": {"tokenize": False, "context": 16000},
            "engine_running": True,
        }

        self.mock_engine_manager.get_current_state.return_value = mock_state
        self.mock_engine_manager.get_current_client.return_value = None

        # Request data for text completion
        request_data = {
            "prompt": "Hello, this is a test prompt with about 100 characters in total.",
            "max_tokens": 200,
        }

        # Select variant
        variant = await self.selector.select_variant(
            model_name=self.model_name,
            request_data=request_data,
            engine_manager=self.mock_engine_manager,
        )

        # Should use fallback estimation
        # Prompt chars: ~100, estimated tokens: 100/4 = 25
        # Total tokens: 25 + 200 = 225
        # Safety margin: max(512, 22.5) = 512
        # Required: 225 + 512 = 737 < 32000
        self.assertEqual(variant["context"], 32000)

    async def test_select_variant_model_not_found(self):
        """Test error handling when model is not found."""
        with self.assertRaises(ValueError) as cm:
            await self.selector.select_variant(
                model_name="non-existent-model",
                request_data={"messages": [{"role": "user", "content": "Hello"}]},
                engine_manager=self.mock_engine_manager,
            )

        self.assertIn("not found", str(cm.exception))

    async def test_select_variant_current_variant_no_tokenization(self):
        """Test variant selection when current variant doesn't support tokenization."""
        # Mock engine manager state with variant that doesn't support tokenization
        mock_state = {
            "model_name": self.model_name,
            "variant_config": {
                "tokenize": False,  # Doesn't support tokenization
                "context": 32000,
            },
            "engine_running": True,
        }

        # Even with a client, tokenization shouldn't be used
        mock_client = AsyncMock()

        self.mock_engine_manager.get_current_state.return_value = mock_state
        self.mock_engine_manager.get_current_client.return_value = mock_client

        # Request data
        request_data = {
            "messages": [
                {"role": "user", "content": "Test message"},
            ],
            "max_tokens": 100,
        }

        # Select variant - should use fallback estimation
        variant = await self.selector.select_variant(
            model_name=self.model_name,
            request_data=request_data,
            engine_manager=self.mock_engine_manager,
        )

        # Should still select a variant (fallback estimation)
        self.assertIsNotNone(variant)
        # Token estimation should NOT have been called
        mock_client.estimate_tokens.assert_not_called()

    async def test_select_variant_tokenization_fails(self):
        """Test fallback when tokenization fails."""
        # Mock engine manager state
        mock_state = {
            "model_name": self.model_name,
            "variant_config": {
                "tokenize": True,
                "context": 32000,
            },
            "engine_running": True,
        }

        mock_client = AsyncMock()
        mock_client.estimate_tokens = AsyncMock(
            side_effect=Exception("Tokenization failed")
        )

        self.mock_engine_manager.get_current_state.return_value = mock_state
        self.mock_engine_manager.get_current_client.return_value = mock_client

        # Request data
        request_data = {
            "messages": [
                {"role": "user", "content": "Test message"},
            ],
            "max_tokens": 100,
        }

        # Should use fallback estimation without raising exception
        variant = await self.selector.select_variant(
            model_name=self.model_name,
            request_data=request_data,
            engine_manager=self.mock_engine_manager,
        )

        # Should still select a variant
        self.assertIsNotNone(variant)
        # Token estimation should have been attempted
        mock_client.estimate_tokens.assert_called_once()

    def test_list_models(self):
        """Test listing all configured model names."""
        models = self.selector.list_models()

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0], self.model_name)

    def test_get_model_info(self):
        """Test getting information about a specific model."""
        model_info = self.selector.get_model_info(self.model_name)

        self.assertEqual(model_info["id"], self.model_name)
        self.assertEqual(model_info["engine"], self.engine_type)
        self.assertEqual(model_info["variants_count"], 2)
        self.assertEqual(model_info["available_context_sizes"], [32000, 64000])
        self.assertTrue(model_info["supports_tokenization"])

        # Check required OpenAI-compatible fields
        self.assertEqual(model_info["object"], "model")
        self.assertEqual(model_info["owned_by"], "llm-gateway")

    def test_get_model_info_not_found(self):
        """Test getting information for non-existent model."""
        with self.assertRaises(ValueError) as cm:
            self.selector.get_model_info("non-existent-model")

        self.assertIn("not found", str(cm.exception))

    def test_get_available_models(self):
        """Test getting detailed information about all configured models."""
        models_info = self.selector.get_available_models()

        self.assertEqual(len(models_info), 1)
        model_info = models_info[0]

        self.assertEqual(model_info["id"], self.model_name)
        self.assertEqual(model_info["variants_count"], 2)

    def test_get_model_engine_type(self):
        """Test getting engine type for a model."""
        engine_type = self.selector.get_model_engine_type(self.model_name)

        self.assertEqual(engine_type, self.engine_type)

    def test_get_model_engine_type_not_found(self):
        """Test getting engine type for non-existent model."""
        with self.assertRaises(ValueError) as cm:
            self.selector.get_model_engine_type("non-existent-model")

        self.assertIn("not found", str(cm.exception))

    def test_get_all_variants(self):
        """Test getting all variants for a model."""
        variants = self.selector.get_all_variants(self.model_name)

        self.assertEqual(len(variants), 2)
        # Should be sorted by context
        self.assertEqual(variants[0]["context"], 32000)
        self.assertEqual(variants[1]["context"], 64000)

    def test_get_all_variants_not_found(self):
        """Test getting variants for non-existent model."""
        with self.assertRaises(ValueError) as cm:
            self.selector.get_all_variants("non-existent-model")

        self.assertIn("not found", str(cm.exception))

    def test_fallback_token_estimation_chat_completion(self):
        """Test fallback token estimation for chat completions."""
        request_data = {
            "messages": [
                {"role": "user", "content": "Hello"},  # 5 chars
                {"role": "assistant", "content": "Hi there!"},  # 9 chars
                {"role": "user", "content": "How are you?"},  # 12 chars
            ],
            "max_tokens": 50,
        }

        tokens = self.selector._fallback_token_estimation(request_data)

        # Total chars: 5 + 9 + 12 = 26
        # Estimated tokens: ceil(26 / 4) = 7
        # Total tokens: 7 + 50 = 57
        self.assertEqual(tokens, 57)

    def test_fallback_token_estimation_text_completion(self):
        """Test fallback token estimation for text completions."""
        request_data = {
            "prompt": "This is a test prompt with many characters for estimation.",
            "max_tokens": 100,
        }

        tokens = self.selector._fallback_token_estimation(request_data)

        # Prompt chars: 60
        # Estimated tokens: ceil(60 / 4) = 15
        # Total tokens: 15 + 100 = 115
        self.assertEqual(tokens, 115)

    def test_fallback_token_estimation_list_prompt(self):
        """Test fallback token estimation with list prompt."""
        request_data = {
            "prompt": ["First prompt", "Second prompt", "Third prompt"],
            "max_tokens": 75,
        }

        tokens = self.selector._fallback_token_estimation(request_data)

        # Total chars: 12 + 13 + 12 = 37
        # Estimated tokens: ceil(37 / 4) = 10
        # Total tokens: 10 + 75 = 85
        self.assertEqual(tokens, 85)

    def test_fallback_token_estimation_empty_messages(self):
        """Test fallback token estimation with empty messages."""
        request_data = {
            "messages": [],
            "max_tokens": 20,
        }

        tokens = self.selector._fallback_token_estimation(request_data)

        # No prompt tokens, just max_tokens
        # Minimum token count is 10, so 20 > 10
        self.assertEqual(tokens, 20)

    def test_fallback_token_estimation_small_prompt(self):
        """Test fallback token estimation with very small prompt."""
        request_data = {
            "prompt": "Hi",  # 2 chars
            "max_tokens": 5,
        }

        tokens = self.selector._fallback_token_estimation(request_data)

        # Estimated tokens: ceil(2 / 4) = 1
        # Total tokens: 1 + 5 = 6
        # But minimum is 10, so should return 10
        self.assertEqual(tokens, 10)

    def test_fallback_token_estimation_no_max_tokens(self):
        """Test fallback token estimation without max_tokens."""
        request_data = {
            "messages": [
                {"role": "user", "content": "Test"},
            ],
        }

        tokens = self.selector._fallback_token_estimation(request_data)

        # Prompt chars: 4, estimated tokens: 1
        # No max_tokens, so total is 1
        # Minimum is 10, so should return 10
        self.assertEqual(tokens, 10)

    def test_parse_models_with_multiple_models(self):
        """Test parsing configuration with multiple models."""
        # Reset mock to handle multiple models
        mock_cfg = MagicMock(spec=python_lua_helper.PyLuaHelper)

        # Mock two models
        def get_table_seq_side_effect(key):
            if key == "models":
                return [1, 2]
            elif key == "models.1.variants":
                return [1]
            elif key == "models.2.variants":
                return [1]
            return []

        mock_cfg.get_table_seq.side_effect = get_table_seq_side_effect

        def get_side_effect(key, default=None):
            if key == "models.1.name":
                return "model-1"
            elif key == "models.1.engine":
                return "llama.cpp"
            elif key == "models.1.variants.1.binary":
                return "/path1"
            elif key == "models.1.variants.1.connect":
                return "http://1:8080"
            elif key == "models.1.variants.1.tokenize":
                return "true"
            elif key == "models.1.variants.1.context":
                return "32000"
            elif key == "models.2.name":
                return "model-2"
            elif key == "models.2.engine":
                return "llama.cpp"
            elif key == "models.2.variants.1.binary":
                return "/path2"
            elif key == "models.2.variants.1.connect":
                return "http://2:8080"
            elif key == "models.2.variants.1.tokenize":
                return "false"
            elif key == "models.2.variants.1.context":
                return "16000"
            return default

        mock_cfg.get.side_effect = get_side_effect

        def get_bool_side_effect(key, default=False):
            if key == "models.1.variants.1.tokenize":
                return True
            elif key == "models.2.variants.1.tokenize":
                return False
            return default

        mock_cfg.get_bool.side_effect = get_bool_side_effect

        def get_int_side_effect(key, default=0):
            if key == "models.1.variants.1.context":
                return 32000
            elif key == "models.2.variants.1.context":
                return 16000
            return default

        mock_cfg.get_int.side_effect = get_int_side_effect

        def get_list_side_effect(key):
            if "args" in key:
                return []
            return []

        mock_cfg.get_list.side_effect = get_list_side_effect

        selector = ModelSelector(mock_cfg)

        # Should have 2 models
        models = selector.list_models()
        self.assertEqual(len(models), 2)
        self.assertIn("model-1", models)
        self.assertIn("model-2", models)

        # Check model 1 supports tokenization
        model1_info = selector.get_model_info("model-1")
        self.assertTrue(model1_info["supports_tokenization"])

        # Check model 2 doesn't support tokenization
        model2_info = selector.get_model_info("model-2")
        self.assertFalse(model2_info["supports_tokenization"])


if __name__ == "__main__":
    unittest.main()
