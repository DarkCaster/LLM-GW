# tests/test_model_selector.py

import unittest
import asyncio
from unittest.mock import AsyncMock
from models.model_selector import ModelSelector
from engine.engine_manager import EngineManager
from engine.engine_client import EngineClient


class MockPyLuaHelper:
    """Mock PyLuaHelper for testing."""

    def __init__(self, models_data):
        """
        Initialize mock with models data.

        Args:
            models_data: List of model dictionaries
        """
        self.models_data = models_data
        self._variables = {}
        self._setup_variables()

    def _setup_variables(self):
        """Setup internal variables from models data."""
        for i, model in enumerate(self.models_data, start=1):
            model_path = f"models.{i}"
            self._variables[f"{model_path}.name"] = model["name"]
            self._variables[f"{model_path}.engine"] = model["engine"]

            for j, variant in enumerate(model["variants"], start=1):
                variant_path = f"{model_path}.variants.{j}"
                self._variables[f"{variant_path}.binary"] = variant["binary"]
                self._variables[f"{variant_path}.connect"] = variant["connect"]
                self._variables[f"{variant_path}.tokenize"] = variant["tokenize"]
                self._variables[f"{variant_path}.context"] = variant["context"]
                for k, arg in enumerate(variant.get("args", []), start=1):
                    self._variables[f"{variant_path}.args.{k}"] = arg

    def get(self, key, default=None):
        """Get variable value."""
        return self._variables.get(key, default)

    def get_int(self, key, default=None):
        """Get variable as integer."""
        value = self._variables.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def get_bool(self, key, default=None):
        """Get variable as boolean."""
        value = self._variables.get(key)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return default

    def get_list(self, key):
        """Get list from table."""
        result = []
        for idx in self.get_table_seq(key):
            result.append(self.get(f"{key}.{idx}"))
        return result

    def get_table_start(self, key):
        """Get table start index."""
        # Check if any elements exist
        for var_key in self._variables:
            if (
                var_key.startswith(f"{key}.")
                and var_key[len(key) + 1 :].split(".")[0].isdigit()
            ):
                return 1
        return 0

    def get_table_end(self, key):
        """Get table end index."""
        max_idx = 0
        prefix = f"{key}."
        for var_key in self._variables:
            if var_key.startswith(prefix):
                # Extract first numeric component after prefix
                remainder = var_key[len(prefix) :]
                first_component = remainder.split(".")[0]
                if first_component.isdigit():
                    max_idx = max(max_idx, int(first_component))
        return max_idx + 1

    def get_table_seq(self, key):
        """Get sequence of table indices."""
        start = self.get_table_start(key)
        end = self.get_table_end(key)
        if start == 0:
            return []
        return list(range(start, end))


class MockEngineClient(EngineClient):
    """Mock EngineClient for testing."""

    async def estimate_tokens(self, request_data: dict) -> int:
        return 100

    def transform_request(self, request_data: dict) -> dict:
        return request_data

    def transform_response(self, response_data: dict) -> dict:
        return response_data

    def get_supported_endpoints(self):
        return ["/v1/chat/completions"]


class TestModelSelector(unittest.TestCase):
    """Test ModelSelector model selection and variant choosing logic."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset EngineManager singleton
        EngineManager._instance = None
        EngineManager._lock = None

    def tearDown(self):
        """Clean up after tests."""
        EngineManager._instance = None
        EngineManager._lock = None

    def test_initialization(self):
        """Test ModelSelector initialization."""
        models_data = [
            {
                "name": "test-model",
                "engine": "llama.cpp",
                "variants": [
                    {
                        "binary": "/usr/bin/llama-server",
                        "connect": "http://localhost:8080",
                        "args": ["-c", "2048"],
                        "tokenize": True,
                        "context": 2048,
                    }
                ],
            }
        ]

        cfg = MockPyLuaHelper(models_data)
        selector = ModelSelector(cfg)

        self.assertIsNotNone(selector)
        self.assertEqual(len(selector.models), 1)
        self.assertIn("test-model", selector.models)

    def test_parse_models_single_model(self):
        """Test parsing a single model from configuration."""
        models_data = [
            {
                "name": "model-a",
                "engine": "llama.cpp",
                "variants": [
                    {
                        "binary": "/usr/bin/llama-server",
                        "connect": "http://localhost:8080",
                        "args": ["-c", "2048"],
                        "tokenize": True,
                        "context": 2048,
                    }
                ],
            }
        ]

        cfg = MockPyLuaHelper(models_data)
        selector = ModelSelector(cfg)

        self.assertEqual(len(selector.models), 1)
        self.assertIn("model-a", selector.models)

        model = selector.models["model-a"]
        self.assertEqual(model["engine"], "llama.cpp")
        self.assertEqual(len(model["variants"]), 1)

        variant = model["variants"][0]
        self.assertEqual(variant["binary"], "/usr/bin/llama-server")
        self.assertEqual(variant["connect"], "http://localhost:8080")
        self.assertEqual(variant["context"], 2048)
        self.assertTrue(variant["tokenize"])

    def test_parse_models_multiple_models(self):
        """Test parsing multiple models from configuration."""
        models_data = [
            {
                "name": "model-a",
                "engine": "llama.cpp",
                "variants": [
                    {
                        "binary": "/usr/bin/llama-server",
                        "connect": "http://localhost:8080",
                        "args": [],
                        "tokenize": True,
                        "context": 2048,
                    }
                ],
            },
            {
                "name": "model-b",
                "engine": "llama.cpp",
                "variants": [
                    {
                        "binary": "/usr/bin/llama-server",
                        "connect": "http://localhost:8081",
                        "args": [],
                        "tokenize": False,
                        "context": 4096,
                    }
                ],
            },
        ]

        cfg = MockPyLuaHelper(models_data)
        selector = ModelSelector(cfg)

        self.assertEqual(len(selector.models), 2)
        self.assertIn("model-a", selector.models)
        self.assertIn("model-b", selector.models)

    def test_parse_models_multiple_variants(self):
        """Test parsing model with multiple variants."""
        models_data = [
            {
                "name": "test-model",
                "engine": "llama.cpp",
                "variants": [
                    {
                        "binary": "/usr/bin/llama-server",
                        "connect": "http://localhost:8080",
                        "args": ["-c", "2048"],
                        "tokenize": True,
                        "context": 2048,
                    },
                    {
                        "binary": "/usr/bin/llama-server",
                        "connect": "http://localhost:8080",
                        "args": ["-c", "4096"],
                        "tokenize": True,
                        "context": 4096,
                    },
                    {
                        "binary": "/usr/bin/llama-server",
                        "connect": "http://localhost:8080",
                        "args": ["-c", "8192"],
                        "tokenize": True,
                        "context": 8192,
                    },
                ],
            }
        ]

        cfg = MockPyLuaHelper(models_data)
        selector = ModelSelector(cfg)

        model = selector.models["test-model"]
        self.assertEqual(len(model["variants"]), 3)

        # Verify variants are sorted by context size
        contexts = [v["context"] for v in model["variants"]]
        self.assertEqual(contexts, sorted(contexts))

    def test_parse_models_sorts_variants_by_context(self):
        """Test that variants are sorted by context size."""
        models_data = [
            {
                "name": "test-model",
                "engine": "llama.cpp",
                "variants": [
                    {
                        "binary": "/usr/bin/llama-server",
                        "connect": "http://localhost:8080",
                        "args": [],
                        "tokenize": True,
                        "context": 8192,
                    },
                    {
                        "binary": "/usr/bin/llama-server",
                        "connect": "http://localhost:8080",
                        "args": [],
                        "tokenize": True,
                        "context": 2048,
                    },
                    {
                        "binary": "/usr/bin/llama-server",
                        "connect": "http://localhost:8080",
                        "args": [],
                        "tokenize": True,
                        "context": 4096,
                    },
                ],
            }
        ]

        cfg = MockPyLuaHelper(models_data)
        selector = ModelSelector(cfg)

        variants = selector.models["test-model"]["variants"]
        contexts = [v["context"] for v in variants]

        # Should be sorted: 2048, 4096, 8192
        self.assertEqual(contexts, [2048, 4096, 8192])

    def test_select_variant_model_not_found(self):
        """Test select_variant raises error for unknown model."""

        async def run_test():
            models_data = []
            cfg = MockPyLuaHelper(models_data)
            selector = ModelSelector(cfg)

            engine_manager = EngineManager()
            request_data = {"messages": [{"role": "user", "content": "Hello"}]}

            with self.assertRaises(ValueError) as context:
                await selector.select_variant(
                    "unknown-model", request_data, engine_manager
                )

            self.assertIn("not found", str(context.exception))

        asyncio.run(run_test())

    def test_select_variant_no_variants(self):
        """Test select_variant raises error when model has no variants."""

        async def run_test():
            models_data = [
                {
                    "name": "test-model",
                    "engine": "llama.cpp",
                    "variants": [],
                }
            ]

            cfg = MockPyLuaHelper(models_data)
            selector = ModelSelector(cfg)

            engine_manager = EngineManager()
            request_data = {"messages": [{"role": "user", "content": "Hello"}]}

            with self.assertRaises(ValueError) as context:
                await selector.select_variant(
                    "test-model", request_data, engine_manager
                )

            self.assertIn("no configured variants", str(context.exception))

        asyncio.run(run_test())

    def test_select_variant_with_tokenization(self):
        """Test select_variant using engine tokenization."""

        async def run_test():
            models_data = [
                {
                    "name": "test-model",
                    "engine": "llama.cpp",
                    "variants": [
                        {
                            "binary": "/usr/bin/llama-server",
                            "connect": "http://localhost:8080",
                            "args": [],
                            "tokenize": True,
                            "context": 2048,
                        },
                        {
                            "binary": "/usr/bin/llama-server",
                            "connect": "http://localhost:8080",
                            "args": [],
                            "tokenize": True,
                            "context": 4096,
                        },
                    ],
                }
            ]

            cfg = MockPyLuaHelper(models_data)
            selector = ModelSelector(cfg)

            engine_manager = EngineManager()

            # Set up current engine with tokenization support
            mock_client = MockEngineClient("http://localhost:8080")
            mock_client.estimate_tokens = AsyncMock(return_value=1500)

            engine_manager.current_model_name = "test-model"
            engine_manager.current_engine_client = mock_client
            engine_manager.current_variant_config = {"tokenize": True}

            request_data = {
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
            }

            variant = await selector.select_variant(
                "test-model", request_data, engine_manager
            )

            # With 1500 estimated + safety margin, should select 2048 context variant
            self.assertEqual(variant["context"], 2048)

        asyncio.run(run_test())

    def test_select_variant_requires_larger_context(self):
        """Test select_variant chooses larger variant when needed."""

        async def run_test():
            models_data = [
                {
                    "name": "test-model",
                    "engine": "llama.cpp",
                    "variants": [
                        {
                            "binary": "/usr/bin/llama-server",
                            "connect": "http://localhost:8080",
                            "args": [],
                            "tokenize": True,
                            "context": 2048,
                        },
                        {
                            "binary": "/usr/bin/llama-server",
                            "connect": "http://localhost:8080",
                            "args": [],
                            "tokenize": True,
                            "context": 4096,
                        },
                        {
                            "binary": "/usr/bin/llama-server",
                            "connect": "http://localhost:8080",
                            "args": [],
                            "tokenize": True,
                            "context": 8192,
                        },
                    ],
                }
            ]

            cfg = MockPyLuaHelper(models_data)
            selector = ModelSelector(cfg)

            engine_manager = EngineManager()

            # Set up tokenization to return large value
            mock_client = MockEngineClient("http://localhost:8080")
            mock_client.estimate_tokens = AsyncMock(return_value=3500)

            engine_manager.current_model_name = "test-model"
            engine_manager.current_engine_client = mock_client
            engine_manager.current_variant_config = {"tokenize": True}

            request_data = {"messages": [{"role": "user", "content": "Hello"}]}

            variant = await selector.select_variant(
                "test-model", request_data, engine_manager
            )

            # With 3500 + safety margin, should select 4096 context variant
            self.assertEqual(variant["context"], 4096)

        asyncio.run(run_test())

    def test_select_variant_context_too_large(self):
        """Test select_variant raises error when no variant is large enough."""

        async def run_test():
            models_data = [
                {
                    "name": "test-model",
                    "engine": "llama.cpp",
                    "variants": [
                        {
                            "binary": "/usr/bin/llama-server",
                            "connect": "http://localhost:8080",
                            "args": [],
                            "tokenize": True,
                            "context": 2048,
                        },
                    ],
                }
            ]

            cfg = MockPyLuaHelper(models_data)
            selector = ModelSelector(cfg)

            engine_manager = EngineManager()

            # Set up tokenization to return very large value
            mock_client = MockEngineClient("http://localhost:8080")
            mock_client.estimate_tokens = AsyncMock(return_value=5000)

            engine_manager.current_model_name = "test-model"
            engine_manager.current_engine_client = mock_client
            engine_manager.current_variant_config = {"tokenize": True}

            request_data = {"messages": [{"role": "user", "content": "Hello"}]}

            with self.assertRaises(ValueError) as context:
                await selector.select_variant(
                    "test-model", request_data, engine_manager
                )

            self.assertIn("largest variant", str(context.exception))

        asyncio.run(run_test())

    def test_select_variant_fallback_estimation(self):
        """Test select_variant uses fallback estimation when tokenization unavailable."""

        async def run_test():
            models_data = [
                {
                    "name": "test-model",
                    "engine": "llama.cpp",
                    "variants": [
                        {
                            "binary": "/usr/bin/llama-server",
                            "connect": "http://localhost:8080",
                            "args": [],
                            "tokenize": True,
                            "context": 4096,
                        },
                    ],
                }
            ]

            cfg = MockPyLuaHelper(models_data)
            selector = ModelSelector(cfg)

            engine_manager = EngineManager()

            # No current engine - should use fallback
            request_data = {
                "messages": [{"role": "user", "content": "Hello world! " * 100}],
                "max_tokens": 200,
            }

            variant = await selector.select_variant(
                "test-model", request_data, engine_manager
            )

            # Should select the variant with fallback estimation
            self.assertEqual(variant["context"], 4096)

        asyncio.run(run_test())

    def test_select_variant_tokenization_fails_fallback(self):
        """Test select_variant falls back when tokenization fails."""

        async def run_test():
            models_data = [
                {
                    "name": "test-model",
                    "engine": "llama.cpp",
                    "variants": [
                        {
                            "binary": "/usr/bin/llama-server",
                            "connect": "http://localhost:8080",
                            "args": [],
                            "tokenize": True,
                            "context": 4096,
                        },
                    ],
                }
            ]

            cfg = MockPyLuaHelper(models_data)
            selector = ModelSelector(cfg)

            engine_manager = EngineManager()

            # Set up tokenization to fail
            mock_client = MockEngineClient("http://localhost:8080")
            mock_client.estimate_tokens = AsyncMock(
                side_effect=Exception("Tokenization failed")
            )

            engine_manager.current_model_name = "test-model"
            engine_manager.current_engine_client = mock_client
            engine_manager.current_variant_config = {"tokenize": True}

            request_data = {
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
            }

            # Should fall back to character-based estimation
            variant = await selector.select_variant(
                "test-model", request_data, engine_manager
            )

            self.assertEqual(variant["context"], 4096)

        asyncio.run(run_test())

    def test_fallback_estimate_chat_messages(self):
        """Test fallback estimation with chat messages."""
        models_data = []
        cfg = MockPyLuaHelper(models_data)
        selector = ModelSelector(cfg)

        request_data = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello there!"},
            ],
            "max_tokens": 100,
        }

        estimate = selector._fallback_estimate(request_data)

        # Roughly: len("You are helpful.Hello there!") / 4 + 100
        # = 30 chars / 4 + 100 = 7 + 100 = 107
        self.assertGreater(estimate, 100)
        self.assertLess(estimate, 200)

    def test_fallback_estimate_text_completion(self):
        """Test fallback estimation with text completion."""
        models_data = []
        cfg = MockPyLuaHelper(models_data)
        selector = ModelSelector(cfg)

        request_data = {
            "prompt": "Once upon a time in a land far away",
            "max_tokens": 50,
        }

        estimate = selector._fallback_estimate(request_data)

        # Roughly: 40 chars / 4 + 50 = 10 + 50 = 60
        self.assertGreater(estimate, 50)
        self.assertLess(estimate, 100)

    def test_fallback_estimate_without_max_tokens(self):
        """Test fallback estimation without max_tokens."""
        models_data = []
        cfg = MockPyLuaHelper(models_data)
        selector = ModelSelector(cfg)

        request_data = {
            "messages": [{"role": "user", "content": "Test message"}],
        }

        estimate = selector._fallback_estimate(request_data)

        # Should be just the prompt tokens (no max_tokens)
        self.assertGreater(estimate, 0)
        self.assertLess(estimate, 50)

    def test_fallback_estimate_list_prompt(self):
        """Test fallback estimation with list prompt."""
        models_data = []
        cfg = MockPyLuaHelper(models_data)
        selector = ModelSelector(cfg)

        request_data = {
            "prompt": ["First prompt", "Second prompt", "Third prompt"],
            "max_tokens": 30,
        }

        estimate = selector._fallback_estimate(request_data)

        # Should count all prompts
        self.assertGreater(estimate, 30)

    def test_get_model_info(self):
        """Test get_model_info returns correct information."""
        models_data = [
            {
                "name": "test-model",
                "engine": "llama.cpp",
                "variants": [
                    {
                        "binary": "/usr/bin/llama-server",
                        "connect": "http://localhost:8080",
                        "args": [],
                        "tokenize": True,
                        "context": 2048,
                    },
                    {
                        "binary": "/usr/bin/llama-server",
                        "connect": "http://localhost:8080",
                        "args": [],
                        "tokenize": True,
                        "context": 4096,
                    },
                ],
            }
        ]

        cfg = MockPyLuaHelper(models_data)
        selector = ModelSelector(cfg)

        info = selector.get_model_info("test-model")

        self.assertEqual(info["name"], "test-model")
        self.assertEqual(info["engine"], "llama.cpp")
        self.assertEqual(info["variants_count"], 2)
        self.assertEqual(info["context_sizes"], [2048, 4096])
        self.assertEqual(info["min_context"], 2048)
        self.assertEqual(info["max_context"], 4096)

    def test_get_model_info_not_found(self):
        """Test get_model_info raises error for unknown model."""
        models_data = []
        cfg = MockPyLuaHelper(models_data)
        selector = ModelSelector(cfg)

        with self.assertRaises(ValueError) as context:
            selector.get_model_info("unknown-model")

        self.assertIn("not found", str(context.exception))

    def test_list_models(self):
        """Test list_models returns all model names."""
        models_data = [
            {
                "name": "model-a",
                "engine": "llama.cpp",
                "variants": [
                    {
                        "binary": "/usr/bin/llama-server",
                        "connect": "http://localhost:8080",
                        "args": [],
                        "tokenize": True,
                        "context": 2048,
                    }
                ],
            },
            {
                "name": "model-b",
                "engine": "llama.cpp",
                "variants": [
                    {
                        "binary": "/usr/bin/llama-server",
                        "connect": "http://localhost:8081",
                        "args": [],
                        "tokenize": True,
                        "context": 4096,
                    }
                ],
            },
        ]

        cfg = MockPyLuaHelper(models_data)
        selector = ModelSelector(cfg)

        models = selector.list_models()

        self.assertEqual(len(models), 2)
        self.assertIn("model-a", models)
        self.assertIn("model-b", models)

    def test_list_models_empty(self):
        """Test list_models with no models configured."""
        models_data = []
        cfg = MockPyLuaHelper(models_data)
        selector = ModelSelector(cfg)

        models = selector.list_models()

        self.assertEqual(len(models), 0)

    def test_safety_margin_calculation(self):
        """Test that safety margin is properly calculated."""

        async def run_test():
            models_data = [
                {
                    "name": "test-model",
                    "engine": "llama.cpp",
                    "variants": [
                        {
                            "binary": "/usr/bin/llama-server",
                            "connect": "http://localhost:8080",
                            "args": [],
                            "tokenize": True,
                            "context": 2048,
                        },
                    ],
                }
            ]

            cfg = MockPyLuaHelper(models_data)
            selector = ModelSelector(cfg)

            engine_manager = EngineManager()

            # Test with small token count - should use minimum 512 margin
            mock_client = MockEngineClient("http://localhost:8080")
            mock_client.estimate_tokens = AsyncMock(return_value=100)

            engine_manager.current_model_name = "test-model"
            engine_manager.current_engine_client = mock_client
            engine_manager.current_variant_config = {"tokenize": True}

            request_data = {"messages": [{"role": "user", "content": "Hi"}]}

            # With 100 tokens + 512 minimum margin = 612, should fit in 2048
            variant = await selector.select_variant(
                "test-model", request_data, engine_manager
            )
            self.assertEqual(variant["context"], 2048)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
