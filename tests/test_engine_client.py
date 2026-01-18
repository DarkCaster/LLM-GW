import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from engine.engine_client import EngineClient
from engine.llamacpp_engine import LlamaCppEngine


class TestEngineClient(unittest.TestCase):
    """Test the base EngineClient abstract class."""

    def test_abstract_base_class(self):
        """Test that EngineClient is an abstract base class."""
        with self.assertRaises(TypeError):
            # Can't instantiate abstract class
            EngineClient(base_url="http://test:8080")

    def test_get_supported_endpoints_not_implemented(self):
        """Test that get_supported_endpoints is abstract."""

        class ConcreteEngineClient(EngineClient):
            async def estimate_tokens(self, request_data: dict) -> int:
                return 0

            def transform_request(self, request_data: dict) -> dict:
                return request_data

            def transform_response(self, response_data: dict) -> dict:
                return response_data

        # Should raise NotImplementedError
        with self.assertRaises(NotImplementedError):
            ConcreteEngineClient(base_url="http://test:8080").get_supported_endpoints()


class TestLlamaCppEngine(unittest.IsolatedAsyncioTestCase):
    """Test the LlamaCppEngine implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_url = "http://test:8080"
        self.engine = LlamaCppEngine(base_url=self.base_url)

    async def test_estimate_tokens_chat_completion(self):
        """Test token estimation for chat completion requests."""
        request_data = {
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm fine, thank you!"},
            ],
            "max_tokens": 100,
        }

        # Mock the tokenization endpoint response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"tokens": [1, 2, 3, 4, 5]})

        with patch.object(
            aiohttp.ClientSession, "post", AsyncMock(return_value=mock_response)
        ):
            tokens = await self.engine.estimate_tokens(request_data)
            self.assertEqual(tokens, 105)  # 5 prompt tokens + 100 max_tokens

    async def test_estimate_tokens_text_completion(self):
        """Test token estimation for text completion requests."""
        request_data = {
            "prompt": "Once upon a time",
            "max_tokens": 50,
        }

        # Mock the tokenization endpoint response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"tokens": [1, 2, 3]})

        with patch.object(
            aiohttp.ClientSession, "post", AsyncMock(return_value=mock_response)
        ):
            tokens = await self.engine.estimate_tokens(request_data)
            self.assertEqual(tokens, 53)  # 3 prompt tokens + 50 max_tokens

    async def test_estimate_tokens_empty_prompt(self):
        """Test token estimation with empty prompt."""
        request_data = {
            "messages": [],
            "max_tokens": 10,
        }

        tokens = await self.engine.estimate_tokens(request_data)
        self.assertEqual(tokens, 10)  # 0 prompt tokens + 10 max_tokens

    async def test_estimate_tokens_tokenization_failure(self):
        """Test token estimation when tokenization endpoint fails."""
        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
        }

        # Mock failed response
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal server error")

        with patch.object(
            aiohttp.ClientSession, "post", AsyncMock(return_value=mock_response)
        ):
            with self.assertRaises(RuntimeError):
                await self.engine.estimate_tokens(request_data)

    async def test_estimate_tokens_timeout(self):
        """Test token estimation when tokenization times out."""
        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
        }

        with patch.object(
            aiohttp.ClientSession, "post", AsyncMock(side_effect=asyncio.TimeoutError())
        ):
            with self.assertRaises(RuntimeError):
                await self.engine.estimate_tokens(request_data)

    def test_transform_request_removes_unsupported_fields(self):
        """Test that unsupported fields are removed from requests."""
        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "logit_bias": {"123": 0.5},
            "logprobs": True,
            "top_logprobs": 5,
            "user": "test_user",
            "model": "test-model",
            "stream": False,
        }

        transformed = self.engine.transform_request(request_data)

        # Check unsupported fields are removed
        self.assertNotIn("logit_bias", transformed)
        self.assertNotIn("logprobs", transformed)
        self.assertNotIn("top_logprobs", transformed)
        self.assertNotIn("user", transformed)

        # Check supported fields remain
        self.assertIn("messages", transformed)
        self.assertIn("model", transformed)
        self.assertIn("stream", transformed)

    def test_transform_response_non_streaming_chat(self):
        """Test response transformation for non-streaming chat completions."""
        response_data = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello there!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        transformed = self.engine.transform_response(response_data)

        # Should have object field added
        self.assertEqual(transformed["object"], "chat.completion")
        # Original structure should be preserved
        self.assertEqual(
            transformed["choices"][0]["message"]["content"], "Hello there!"
        )

    def test_transform_response_non_streaming_text(self):
        """Test response transformation for non-streaming text completions."""
        response_data = {
            "choices": [
                {
                    "index": 0,
                    "text": "Once upon a time",
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 20},
        }

        transformed = self.engine.transform_response(response_data)

        # Should have object field for text completion
        self.assertEqual(transformed["object"], "text_completion")
        # Text field should be preserved
        self.assertEqual(transformed["choices"][0]["text"], "Once upon a time")

    def test_transform_response_streaming_chat(self):
        """Test response transformation for streaming chat completions."""
        response_data = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "finish_reason": None,
                }
            ],
            "stream": True,
        }

        transformed = self.engine.transform_response(response_data)

        # Should have delta structure
        self.assertIn("delta", transformed["choices"][0])
        self.assertEqual(transformed["choices"][0]["delta"]["content"], "Hello")

    def test_transform_response_streaming_text(self):
        """Test response transformation for streaming text completions."""
        response_data = {
            "choices": [
                {
                    "index": 0,
                    "text": "Once",
                    "finish_reason": None,
                }
            ],
            "stream": True,
        }

        transformed = self.engine.transform_response(response_data)

        # Should have text field
        self.assertIn("text", transformed["choices"][0])
        self.assertEqual(transformed["choices"][0]["text"], "Once")

    def test_get_supported_endpoints(self):
        """Test that supported endpoints are correctly listed."""
        endpoints = self.engine.get_supported_endpoints()

        expected_endpoints = [
            "/v1/chat/completions",
            "/v1/completions",
            "/tokenize",
            "/health",
        ]

        self.assertEqual(set(endpoints), set(expected_endpoints))

    async def test_check_health_success(self):
        """Test health check with successful response."""
        # Mock successful health endpoint response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "ok"})

        with patch.object(
            aiohttp.ClientSession, "get", AsyncMock(return_value=mock_response)
        ):
            is_healthy = await self.engine.check_health()
            self.assertTrue(is_healthy)

    async def test_check_health_failure(self):
        """Test health check with failed response."""
        # Mock failed health endpoint response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "not ok"})

        with patch.object(
            aiohttp.ClientSession, "get", AsyncMock(return_value=mock_response)
        ):
            is_healthy = await self.engine.check_health()
            self.assertFalse(is_healthy)

    async def test_check_health_timeout(self):
        """Test health check when server times out."""
        with patch.object(
            aiohttp.ClientSession, "get", AsyncMock(side_effect=asyncio.TimeoutError())
        ):
            # Should fall back to generic health check
            with patch.object(
                EngineClient, "check_health", AsyncMock(return_value=False)
            ):
                is_healthy = await self.engine.check_health()
                self.assertFalse(is_healthy)

    async def test_forward_request_success(self):
        """Test successful request forwarding."""
        endpoint = "/v1/chat/completions"
        request_data = {"messages": [{"role": "user", "content": "Hello"}]}
        timeout = 30.0

        # Mock the transformed request and response
        transformed_request = {"messages": [{"role": "user", "content": "Hello"}]}
        mock_response = MagicMock(spec=aiohttp.ClientResponse)

        with patch.object(
            self.engine, "transform_request", return_value=transformed_request
        ):
            with patch.object(
                aiohttp.ClientSession, "post", AsyncMock(return_value=mock_response)
            ):
                async with aiohttp.ClientSession() as session:
                    response = await self.engine.forward_request(
                        session=session,
                        endpoint=endpoint,
                        request_data=request_data,
                        timeout=timeout,
                    )
                    self.assertEqual(response, mock_response)

    async def test_forward_request_unsupported_endpoint(self):
        """Test request forwarding with unsupported endpoint."""
        endpoint = "/unsupported"
        request_data = {"test": "data"}

        with self.assertRaises(ValueError):
            async with aiohttp.ClientSession() as session:
                await self.engine.forward_request(
                    session=session,
                    endpoint=endpoint,
                    request_data=request_data,
                )

    async def test_forward_request_http_error(self):
        """Test request forwarding when HTTP request fails."""
        endpoint = "/v1/chat/completions"
        request_data = {"messages": [{"role": "user", "content": "Hello"}]}

        with patch.object(
            aiohttp.ClientSession, "post", AsyncMock(side_effect=aiohttp.ClientError())
        ):
            with self.assertRaises(aiohttp.ClientError):
                async with aiohttp.ClientSession() as session:
                    await self.engine.forward_request(
                        session=session,
                        endpoint=endpoint,
                        request_data=request_data,
                    )

    def test_init_with_custom_logger(self):
        """Test initialization with custom logger."""
        custom_logger = MagicMock()
        engine = LlamaCppEngine(base_url=self.base_url, logger=custom_logger)
        self.assertEqual(engine.logger, custom_logger)

    def test_init_without_logger(self):
        """Test initialization without custom logger."""
        engine = LlamaCppEngine(base_url=self.base_url)
        self.assertEqual(engine.logger.name, "LlamaCppEngine")


if __name__ == "__main__":
    unittest.main()
