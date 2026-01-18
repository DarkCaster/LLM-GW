# tests/test_engine_client.py

import unittest
import asyncio
import aiohttp
from unittest.mock import AsyncMock, patch
from engine.engine_client import EngineClient
from engine.llamacpp_engine import LlamaCppEngine


class MockEngineClient(EngineClient):
    """Concrete implementation of abstract EngineClient for testing."""

    async def estimate_tokens(self, request_data: dict) -> int:
        """Mock implementation."""
        return 100

    def transform_request(self, request_data: dict) -> dict:
        """Mock implementation."""
        return request_data

    def transform_response(self, response_data: dict) -> dict:
        """Mock implementation."""
        return response_data

    def get_supported_endpoints(self):
        """Mock implementation."""
        return ["/v1/chat/completions", "/v1/completions"]


class TestEngineClient(unittest.TestCase):
    """Test base EngineClient abstract class."""

    def test_abstract_base_class_cannot_instantiate(self):
        """Test that EngineClient cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            EngineClient("http://localhost:8080")

    def test_mock_engine_client_can_instantiate(self):
        """Test that concrete implementation can be instantiated."""
        client = MockEngineClient("http://localhost:8080")
        self.assertIsNotNone(client)
        self.assertEqual(client.base_url, "http://localhost:8080")

    def test_base_url_strips_trailing_slash(self):
        """Test that base URL trailing slash is removed."""
        client = MockEngineClient("http://localhost:8080/")
        self.assertEqual(client.base_url, "http://localhost:8080")

    def test_abstract_methods_must_be_implemented(self):
        """Test that all abstract methods must be implemented."""
        client = MockEngineClient("http://localhost:8080")

        # These should all be callable without errors
        asyncio.run(client.estimate_tokens({}))
        client.transform_request({})
        client.transform_response({})
        client.get_supported_endpoints()


class TestLlamaCppEngine(unittest.TestCase):
    """Test LlamaCppEngine implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_url = "http://localhost:8080"
        self.engine = LlamaCppEngine(self.base_url)

    def test_initialization(self):
        """Test LlamaCppEngine initialization."""
        self.assertEqual(self.engine.base_url, self.base_url)
        self.assertIsNotNone(self.engine.logger)

    def test_get_supported_endpoints(self):
        """Test that LlamaCppEngine returns correct supported endpoints."""
        endpoints = self.engine.get_supported_endpoints()
        self.assertIn("/v1/chat/completions", endpoints)
        self.assertIn("/v1/completions", endpoints)
        self.assertEqual(len(endpoints), 2)

    def test_estimate_tokens_chat_completion(self):
        """Test token estimation for chat completion requests."""

        async def run_test():
            request_data = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, how are you?"},
                ],
                "max_tokens": 100,
            }

            # Mock the tokenization response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={"tokens": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
            )

            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            with patch("aiohttp.ClientSession", return_value=mock_session):
                tokens = await self.engine.estimate_tokens(request_data)

            # Should be 10 (prompt) + 100 (max_tokens) = 110
            self.assertEqual(tokens, 110)

        asyncio.run(run_test())

    def test_estimate_tokens_text_completion(self):
        """Test token estimation for text completion requests."""

        async def run_test():
            request_data = {
                "prompt": "Once upon a time in a land far away",
                "max_tokens": 50,
            }

            # Mock the tokenization response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"tokens": [1, 2, 3, 4, 5]})

            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            with patch("aiohttp.ClientSession", return_value=mock_session):
                tokens = await self.engine.estimate_tokens(request_data)

            # Should be 5 (prompt) + 50 (max_tokens) = 55
            self.assertEqual(tokens, 55)

        asyncio.run(run_test())

    def test_estimate_tokens_list_prompt(self):
        """Test token estimation with list-based prompt."""

        async def run_test():
            request_data = {
                "prompt": ["First prompt", "Second prompt"],
                "max_tokens": 20,
            }

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"tokens": [1, 2, 3]})

            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            with patch("aiohttp.ClientSession", return_value=mock_session):
                tokens = await self.engine.estimate_tokens(request_data)

            # Should be 3 (prompt) + 20 (max_tokens) = 23
            self.assertEqual(tokens, 23)

        asyncio.run(run_test())

    def test_estimate_tokens_without_max_tokens(self):
        """Test token estimation without max_tokens field."""

        async def run_test():
            request_data = {
                "messages": [{"role": "user", "content": "Hello"}],
            }

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"tokens": [1, 2, 3, 4]})

            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            with patch("aiohttp.ClientSession", return_value=mock_session):
                tokens = await self.engine.estimate_tokens(request_data)

            # Should be 4 (prompt) + 0 (no max_tokens) = 4
            self.assertEqual(tokens, 4)

        asyncio.run(run_test())

    def test_estimate_tokens_failure(self):
        """Test token estimation error handling."""

        async def run_test():
            request_data = {"messages": [{"role": "user", "content": "Test"}]}

            # Mock failed tokenization response
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Internal Server Error")

            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            with patch("aiohttp.ClientSession", return_value=mock_session):
                with self.assertRaises(Exception) as context:
                    await self.engine.estimate_tokens(request_data)

                self.assertIn("Tokenization request failed", str(context.exception))

        asyncio.run(run_test())

    def test_estimate_tokens_network_error(self):
        """Test token estimation with network error."""

        async def run_test():
            request_data = {"messages": [{"role": "user", "content": "Test"}]}

            mock_session = AsyncMock()
            mock_session.post = AsyncMock(
                side_effect=aiohttp.ClientError("Connection failed")
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            with patch("aiohttp.ClientSession", return_value=mock_session):
                with self.assertRaises(Exception) as context:
                    await self.engine.estimate_tokens(request_data)

                self.assertIn("Token estimation failed", str(context.exception))

        asyncio.run(run_test())

    def test_transform_request_removes_unsupported_fields(self):
        """Test that transform_request removes unsupported fields."""
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "user": "test-user",
            "logit_bias": {"50256": -100},
            "functions": [{"name": "test"}],
            "function_call": "auto",
            "tools": [{"type": "function"}],
            "tool_choice": "auto",
        }

        transformed = self.engine.transform_request(request_data)

        # Should keep basic fields
        self.assertIn("model", transformed)
        self.assertIn("messages", transformed)
        self.assertIn("temperature", transformed)

        # Should remove unsupported fields
        self.assertNotIn("user", transformed)
        self.assertNotIn("logit_bias", transformed)
        self.assertNotIn("functions", transformed)
        self.assertNotIn("function_call", transformed)
        self.assertNotIn("tools", transformed)
        self.assertNotIn("tool_choice", transformed)

    def test_transform_request_does_not_modify_original(self):
        """Test that transform_request doesn't modify the original dict."""
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "user": "test-user",
        }

        original_keys = set(request_data.keys())
        transformed = self.engine.transform_request(request_data)

        # Original should be unchanged
        self.assertEqual(set(request_data.keys()), original_keys)
        self.assertIn("user", request_data)

        # Transformed should have removed the field
        self.assertNotIn("user", transformed)

    def test_transform_request_with_no_unsupported_fields(self):
        """Test transform_request with only supported fields."""
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 100,
        }

        transformed = self.engine.transform_request(request_data)

        # All fields should be preserved
        self.assertEqual(set(request_data.keys()), set(transformed.keys()))

    def test_transform_response_passthrough(self):
        """Test that transform_response passes through data."""
        response_data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
        }

        transformed = self.engine.transform_response(response_data)

        # Should be identical (llama.cpp is already OpenAI-compatible)
        self.assertEqual(transformed, response_data)

    def test_check_health_success_via_health_endpoint(self):
        """Test health check success using /health endpoint."""

        async def run_test():
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            with patch("aiohttp.ClientSession", return_value=mock_session):
                is_healthy = await self.engine.check_health(timeout=5.0)

            self.assertTrue(is_healthy)

        asyncio.run(run_test())

    def test_check_health_fallback_to_models_endpoint(self):
        """Test health check fallback to /v1/models endpoint."""

        async def run_test():
            # First call to /health fails, second call to /v1/models succeeds
            health_response = AsyncMock()
            health_response.status = 404
            health_response.__aenter__ = AsyncMock(return_value=health_response)
            health_response.__aexit__ = AsyncMock(return_value=None)

            models_response = AsyncMock()
            models_response.status = 200
            models_response.__aenter__ = AsyncMock(return_value=models_response)
            models_response.__aexit__ = AsyncMock(return_value=None)

            mock_session = AsyncMock()
            mock_session.get = AsyncMock(side_effect=[health_response, models_response])
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            with patch("aiohttp.ClientSession", return_value=mock_session):
                is_healthy = await self.engine.check_health(timeout=5.0)

            self.assertTrue(is_healthy)

        asyncio.run(run_test())

    def test_check_health_failure_both_endpoints(self):
        """Test health check failure when both endpoints fail."""

        async def run_test():
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            with patch("aiohttp.ClientSession", return_value=mock_session):
                is_healthy = await self.engine.check_health(timeout=5.0)

            self.assertFalse(is_healthy)

        asyncio.run(run_test())

    def test_check_health_timeout(self):
        """Test health check with timeout."""

        async def run_test():
            mock_session = AsyncMock()
            mock_session.get = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            with patch("aiohttp.ClientSession", return_value=mock_session):
                is_healthy = await self.engine.check_health(timeout=1.0)

            self.assertFalse(is_healthy)

        asyncio.run(run_test())

    def test_check_health_network_error(self):
        """Test health check with network error."""

        async def run_test():
            mock_session = AsyncMock()
            mock_session.get = AsyncMock(
                side_effect=aiohttp.ClientError("Connection refused")
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            with patch("aiohttp.ClientSession", return_value=mock_session):
                is_healthy = await self.engine.check_health(timeout=5.0)

            self.assertFalse(is_healthy)

        asyncio.run(run_test())

    def test_forward_request(self):
        """Test forward_request method."""

        async def run_test():
            request_data = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            }

            mock_response = AsyncMock()
            mock_response.status = 200

            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)

            response = await self.engine.forward_request(
                mock_session, "/v1/chat/completions", request_data
            )

            self.assertEqual(response, mock_response)
            mock_session.post.assert_called_once()

            # Verify URL construction
            call_args = mock_session.post.call_args
            self.assertEqual(call_args[0][0], f"{self.base_url}/v1/chat/completions")

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
