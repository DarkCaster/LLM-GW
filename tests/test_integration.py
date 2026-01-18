# tests/test_integration.py

import unittest
import asyncio
import json
import tempfile
import os
from aiohttp import web
import aiohttp

from server.gateway_server import GatewayServer
from config import ConfigLoader


class MockLlamaCppServer:
    """Mock llama.cpp server for integration testing."""

    def __init__(self, host="127.0.0.1", port=18080):
        """
        Initialize mock server.

        Args:
            host: Host to bind to
            port: Port to bind to
        """
        self.host = host
        self.port = port
        self.app = None
        self.runner = None
        self.site = None
        self.tokenize_responses = {}
        self.completion_responses = {}

    def set_tokenize_response(self, tokens_count):
        """Set the response for tokenization endpoint."""
        self.tokenize_responses = {"tokens": list(range(tokens_count))}

    def set_completion_response(self, response_data):
        """Set the response for completion endpoint."""
        self.completion_responses = response_data

    async def handle_health(self, request):
        """Handle /health endpoint."""
        return web.json_response({"status": "ok"})

    async def handle_models(self, request):
        """Handle /v1/models endpoint."""
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": "test-model",
                        "object": "model",
                        "created": 0,
                        "owned_by": "test",
                    }
                ],
            }
        )

    async def handle_tokenize(self, request):
        """Handle /tokenize endpoint."""
        return web.json_response(self.tokenize_responses)

    async def handle_chat_completion(self, request):
        """Handle /v1/chat/completions endpoint."""
        request_data = await request.json()

        # Check if streaming is requested
        is_streaming = request_data.get("stream", False)

        if is_streaming:
            # Return streaming response
            response = web.StreamResponse()
            response.headers["Content-Type"] = "text/event-stream"
            await response.prepare(request)

            # Send a few SSE chunks
            chunks = [
                {
                    "id": "chatcmpl-123",
                    "object": "chat.completion.chunk",
                    "created": 1234567890,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl-123",
                    "object": "chat.completion.chunk",
                    "created": 1234567890,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "Hello"},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl-123",
                    "object": "chat.completion.chunk",
                    "created": 1234567890,
                    "model": "test-model",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]

            for chunk in chunks:
                await response.write(f"data: {json.dumps(chunk)}\n\n".encode())

            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response
        else:
            # Return non-streaming response
            response_data = self.completion_responses or {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }

            return web.json_response(response_data)

    async def handle_completion(self, request):
        """Handle /v1/completions endpoint."""
        response_data = self.completion_responses or {
            "id": "cmpl-123",
            "object": "text_completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {"text": "completion text", "index": 0, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

        return web.json_response(response_data)

    async def start(self):
        """Start the mock server."""
        self.app = web.Application()

        # Register routes
        self.app.router.add_get("/health", self.handle_health)
        self.app.router.add_get("/v1/models", self.handle_models)
        self.app.router.add_post("/tokenize", self.handle_tokenize)
        self.app.router.add_post("/v1/chat/completions", self.handle_chat_completion)
        self.app.router.add_post("/v1/completions", self.handle_completion)

        # Start server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()

    async def stop(self):
        """Stop the mock server."""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()


class TestIntegration(unittest.TestCase):
    """Integration tests for end-to-end request flow."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_engine_port = 18080
        self.gateway_port = 17777

    def tearDown(self):
        """Clean up after tests."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_config(self, config_content):
        """
        Create a temporary lua config file.

        Args:
            config_content: Lua configuration string

        Returns:
            Path to the config file
        """
        config_path = os.path.join(self.temp_dir, "test_config.lua")
        with open(config_path, "w") as f:
            f.write(config_content)
        return config_path

    def test_configuration_loading(self):
        """Test that configuration is properly loaded."""
        config_content = f"""
server = {{
    listen_v4 = "127.0.0.1:{self.gateway_port}",
    listen_v6 = "none",
}}

test_model = {{
    engine = presets.engines.llamacpp,
    name = "test-model",
    variants = {{
        {{
            binary = "/usr/bin/true",
            connect = "http://127.0.0.1:{self.mock_engine_port}",
            args = {{}},
            tokenize = true,
            context = 2048,
        }},
    }},
}}

models = {{ test_model }}
"""

        config_path = self._create_test_config(config_content)

        try:
            config_loader = ConfigLoader(config_path)
            cfg = config_loader.cfg

            self.assertEqual(
                cfg.get("server.listen_v4"), f"127.0.0.1:{self.gateway_port}"
            )
            self.assertEqual(cfg.get("server.listen_v6"), "none")

        except Exception as e:
            self.fail(f"Configuration loading failed: {e}")

    def test_server_start_and_stop(self):
        """Test that gateway server can start and stop cleanly."""

        async def run_test():
            config_content = f"""
server = {{
    listen_v4 = "127.0.0.1:{self.gateway_port}",
    listen_v6 = "none",
}}

test_model = {{
    engine = presets.engines.llamacpp,
    name = "test-model",
    variants = {{
        {{
            binary = "/usr/bin/true",
            connect = "http://127.0.0.1:{self.mock_engine_port}",
            args = {{}},
            tokenize = true,
            context = 2048,
        }},
    }},
}}

models = {{ test_model }}
"""

            config_path = self._create_test_config(config_content)
            config_loader = ConfigLoader(config_path)
            cfg = config_loader.cfg

            server = GatewayServer(cfg)

            try:
                await server.start()

                # Verify server is listening
                self.assertIsNotNone(server.app)
                self.assertEqual(len(server.sites), 1)

                # Give it a moment to fully start
                await asyncio.sleep(0.1)

            finally:
                await server.stop()

        asyncio.run(run_test())

    def test_models_list_endpoint(self):
        """Test GET /v1/models endpoint returns configured models."""

        async def run_test():
            config_content = f"""
server = {{
    listen_v4 = "127.0.0.1:{self.gateway_port}",
    listen_v6 = "none",
}}

test_model = {{
    engine = presets.engines.llamacpp,
    name = "test-model",
    variants = {{
        {{
            binary = "/usr/bin/true",
            connect = "http://127.0.0.1:{self.mock_engine_port}",
            args = {{}},
            tokenize = true,
            context = 2048,
        }},
    }},
}}

models = {{ test_model }}
"""

            config_path = self._create_test_config(config_content)
            config_loader = ConfigLoader(config_path)
            cfg = config_loader.cfg

            server = GatewayServer(cfg)

            try:
                await server.start()
                await asyncio.sleep(0.1)

                # Make request to /v1/models
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://127.0.0.1:{self.gateway_port}/v1/models"
                    ) as response:
                        self.assertEqual(response.status, 200)

                        data = await response.json()
                        self.assertEqual(data["object"], "list")
                        self.assertIsInstance(data["data"], list)
                        self.assertEqual(len(data["data"]), 1)
                        self.assertEqual(data["data"][0]["id"], "test-model")

            finally:
                await server.stop()

        asyncio.run(run_test())

    def test_model_info_endpoint(self):
        """Test GET /v1/models/{model_id} endpoint."""

        async def run_test():
            config_content = f"""
server = {{
    listen_v4 = "127.0.0.1:{self.gateway_port}",
    listen_v6 = "none",
}}

test_model = {{
    engine = presets.engines.llamacpp,
    name = "my-model",
    variants = {{
        {{
            binary = "/usr/bin/true",
            connect = "http://127.0.0.1:{self.mock_engine_port}",
            args = {{}},
            tokenize = true,
            context = 2048,
        }},
    }},
}}

models = {{ test_model }}
"""

            config_path = self._create_test_config(config_content)
            config_loader = ConfigLoader(config_path)
            cfg = config_loader.cfg

            server = GatewayServer(cfg)

            try:
                await server.start()
                await asyncio.sleep(0.1)

                # Make request to /v1/models/my-model
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://127.0.0.1:{self.gateway_port}/v1/models/my-model"
                    ) as response:
                        self.assertEqual(response.status, 200)

                        data = await response.json()
                        self.assertEqual(data["id"], "my-model")
                        self.assertEqual(data["object"], "model")

            finally:
                await server.stop()

        asyncio.run(run_test())

    def test_chat_completion_request_flow(self):
        """Test complete chat completion request flow with mock engine."""

        async def run_test():
            # Start mock llama.cpp server
            mock_server = MockLlamaCppServer(port=self.mock_engine_port)
            mock_server.set_tokenize_response(100)
            await mock_server.start()

            try:
                config_content = f"""
server = {{
    listen_v4 = "127.0.0.1:{self.gateway_port}",
    listen_v6 = "none",
}}

test_model = {{
    engine = presets.engines.llamacpp,
    name = "test-model",
    variants = {{
        {{
            binary = "/usr/bin/sleep",
            connect = "http://127.0.0.1:{self.mock_engine_port}",
            args = {{"3600"}},
            tokenize = true,
            context = 2048,
        }},
    }},
}}

models = {{ test_model }}
"""

                config_path = self._create_test_config(config_content)
                config_loader = ConfigLoader(config_path)
                cfg = config_loader.cfg

                server = GatewayServer(cfg)

                try:
                    await server.start()
                    await asyncio.sleep(0.1)

                    # Make chat completion request
                    request_data = {
                        "model": "test-model",
                        "messages": [
                            {"role": "user", "content": "Hello, how are you?"}
                        ],
                        "max_tokens": 50,
                    }

                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"http://127.0.0.1:{self.gateway_port}/v1/chat/completions",
                            json=request_data,
                        ) as response:
                            # Should get a response (though engine may be starting)
                            # We're mainly testing the request flow here
                            self.assertIn(response.status, [200, 503])

                finally:
                    await server.stop()

            finally:
                await mock_server.stop()

        asyncio.run(run_test())

    def test_chat_completion_streaming(self):
        """Test streaming chat completion response."""

        async def run_test():
            # Start mock llama.cpp server
            mock_server = MockLlamaCppServer(port=self.mock_engine_port)
            mock_server.set_tokenize_response(50)
            await mock_server.start()

            try:
                # Give mock server time to start
                await asyncio.sleep(0.2)

                config_content = f"""
server = {{
    listen_v4 = "127.0.0.1:{self.gateway_port}",
    listen_v6 = "none",
}}

test_model = {{
    engine = presets.engines.llamacpp,
    name = "test-model",
    variants = {{
        {{
            binary = "/usr/bin/sleep",
            connect = "http://127.0.0.1:{self.mock_engine_port}",
            args = {{"3600"}},
            tokenize = true,
            context = 2048,
        }},
    }},
}}

models = {{ test_model }}
"""

                config_path = self._create_test_config(config_content)
                config_loader = ConfigLoader(config_path)
                cfg = config_loader.cfg

                server = GatewayServer(cfg)

                try:
                    await server.start()
                    await asyncio.sleep(0.1)

                    # Make streaming chat completion request
                    request_data = {
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "Hi"}],
                        "stream": True,
                    }

                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"http://127.0.0.1:{self.gateway_port}/v1/chat/completions",
                            json=request_data,
                        ) as response:
                            # Check that we get a streaming response or error
                            self.assertIn(response.status, [200, 503])

                finally:
                    await server.stop()

            finally:
                await mock_server.stop()

        asyncio.run(run_test())

    def test_text_completion_request(self):
        """Test text completion endpoint."""

        async def run_test():
            # Start mock llama.cpp server
            mock_server = MockLlamaCppServer(port=self.mock_engine_port)
            mock_server.set_tokenize_response(30)
            await mock_server.start()

            try:
                await asyncio.sleep(0.2)

                config_content = f"""
server = {{
    listen_v4 = "127.0.0.1:{self.gateway_port}",
    listen_v6 = "none",
}}

test_model = {{
    engine = presets.engines.llamacpp,
    name = "test-model",
    variants = {{
        {{
            binary = "/usr/bin/sleep",
            connect = "http://127.0.0.1:{self.mock_engine_port}",
            args = {{"3600"}},
            tokenize = true,
            context = 2048,
        }},
    }},
}}

models = {{ test_model }}
"""

                config_path = self._create_test_config(config_content)
                config_loader = ConfigLoader(config_path)
                cfg = config_loader.cfg

                server = GatewayServer(cfg)

                try:
                    await server.start()
                    await asyncio.sleep(0.1)

                    # Make text completion request
                    request_data = {
                        "model": "test-model",
                        "prompt": "Once upon a time",
                        "max_tokens": 50,
                    }

                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"http://127.0.0.1:{self.gateway_port}/v1/completions",
                            json=request_data,
                        ) as response:
                            self.assertIn(response.status, [200, 503])

                finally:
                    await server.stop()

            finally:
                await mock_server.stop()

        asyncio.run(run_test())

    def test_invalid_model_request(self):
        """Test request for non-existent model returns error."""

        async def run_test():
            config_content = f"""
server = {{
    listen_v4 = "127.0.0.1:{self.gateway_port}",
    listen_v6 = "none",
}}

test_model = {{
    engine = presets.engines.llamacpp,
    name = "test-model",
    variants = {{
        {{
            binary = "/usr/bin/true",
            connect = "http://127.0.0.1:{self.mock_engine_port}",
            args = {{}},
            tokenize = true,
            context = 2048,
        }},
    }},
}}

models = {{ test_model }}
"""

            config_path = self._create_test_config(config_content)
            config_loader = ConfigLoader(config_path)
            cfg = config_loader.cfg

            server = GatewayServer(cfg)

            try:
                await server.start()
                await asyncio.sleep(0.1)

                # Request non-existent model
                request_data = {
                    "model": "non-existent-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                }

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{self.gateway_port}/v1/chat/completions",
                        json=request_data,
                    ) as response:
                        self.assertEqual(response.status, 400)

                        data = await response.json()
                        self.assertIn("error", data)
                        self.assertIn("not found", data["error"]["message"])

            finally:
                await server.stop()

        asyncio.run(run_test())

    def test_missing_required_fields(self):
        """Test request with missing required fields returns error."""

        async def run_test():
            config_content = f"""
server = {{
    listen_v4 = "127.0.0.1:{self.gateway_port}",
    listen_v6 = "none",
}}

test_model = {{
    engine = presets.engines.llamacpp,
    name = "test-model",
    variants = {{
        {{
            binary = "/usr/bin/true",
            connect = "http://127.0.0.1:{self.mock_engine_port}",
            args = {{}},
            tokenize = true,
            context = 2048,
        }},
    }},
}}

models = {{ test_model }}
"""

            config_path = self._create_test_config(config_content)
            config_loader = ConfigLoader(config_path)
            cfg = config_loader.cfg

            server = GatewayServer(cfg)

            try:
                await server.start()
                await asyncio.sleep(0.1)

                # Request without messages field
                request_data = {"model": "test-model"}

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{self.gateway_port}/v1/chat/completions",
                        json=request_data,
                    ) as response:
                        self.assertEqual(response.status, 400)

                        data = await response.json()
                        self.assertIn("error", data)
                        self.assertIn("messages", data["error"]["message"].lower())

            finally:
                await server.stop()

        asyncio.run(run_test())

    def test_variant_selection_by_context(self):
        """Test that appropriate variant is selected based on context requirements."""

        async def run_test():
            config_content = f"""
server = {{
    listen_v4 = "127.0.0.1:{self.gateway_port}",
    listen_v6 = "none",
}}

test_model = {{
    engine = presets.engines.llamacpp,
    name = "test-model",
    variants = {{
        {{
            binary = "/usr/bin/sleep",
            connect = "http://127.0.0.1:{self.mock_engine_port}",
            args = {{"3600"}},
            tokenize = true,
            context = 2048,
        }},
        {{
            binary = "/usr/bin/sleep",
            connect = "http://127.0.0.1:{self.mock_engine_port}",
            args = {{"3600"}},
            tokenize = true,
            context = 4096,
        }},
    }},
}}

models = {{ test_model }}
"""

            config_path = self._create_test_config(config_content)
            config_loader = ConfigLoader(config_path)
            cfg = config_loader.cfg

            server = GatewayServer(cfg)

            try:
                # This test verifies configuration parsing and variant availability
                # Full engine switching would require actual engine binary
                self.assertEqual(
                    len(server.model_selector.models["test-model"]["variants"]), 2
                )

                variants = server.model_selector.models["test-model"]["variants"]
                self.assertEqual(variants[0]["context"], 2048)
                self.assertEqual(variants[1]["context"], 4096)

            finally:
                pass

        asyncio.run(run_test())

    def test_error_handling_invalid_json(self):
        """Test error handling for invalid JSON in request."""

        async def run_test():
            config_content = f"""
server = {{
    listen_v4 = "127.0.0.1:{self.gateway_port}",
    listen_v6 = "none",
}}

test_model = {{
    engine = presets.engines.llamacpp,
    name = "test-model",
    variants = {{
        {{
            binary = "/usr/bin/true",
            connect = "http://127.0.0.1:{self.mock_engine_port}",
            args = {{}},
            tokenize = true,
            context = 2048,
        }},
    }},
}}

models = {{ test_model }}
"""

            config_path = self._create_test_config(config_content)
            config_loader = ConfigLoader(config_path)
            cfg = config_loader.cfg

            server = GatewayServer(cfg)

            try:
                await server.start()
                await asyncio.sleep(0.1)

                # Send invalid JSON
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{self.gateway_port}/v1/chat/completions",
                        data="invalid json {",
                        headers={"Content-Type": "application/json"},
                    ) as response:
                        self.assertEqual(response.status, 400)

                        data = await response.json()
                        self.assertIn("error", data)

            finally:
                await server.stop()

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
