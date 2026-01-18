# tests/test_engine_manager.py

import unittest
import asyncio
from unittest.mock import AsyncMock, patch
from engine.engine_manager import EngineManager
from engine.engine_client import EngineClient
from engine.engine_process import EngineProcess


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


class TestEngineManager(unittest.TestCase):
    """Test EngineManager coordination logic."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset singleton instance before each test
        EngineManager._instance = None
        EngineManager._lock = None

    def tearDown(self):
        """Clean up after tests."""
        # Reset singleton
        EngineManager._instance = None
        EngineManager._lock = None

    def test_singleton_pattern(self):
        """Test that EngineManager follows singleton pattern."""
        manager1 = EngineManager()
        manager2 = EngineManager()

        self.assertIs(manager1, manager2)

    def test_initialization(self):
        """Test EngineManager initialization."""
        manager = EngineManager()

        self.assertIsNone(manager.current_engine_process)
        self.assertIsNone(manager.current_engine_client)
        self.assertIsNone(manager.current_variant_config)
        self.assertIsNone(manager.current_model_name)
        self.assertIsNotNone(EngineManager._lock)

    def test_get_current_client_none(self):
        """Test get_current_client when no engine is running."""
        manager = EngineManager()

        client = manager.get_current_client()
        self.assertIsNone(client)

    def test_get_current_client_with_engine(self):
        """Test get_current_client when engine is running."""

        async def run_test():
            manager = EngineManager()

            # Set up a mock client
            mock_client = MockEngineClient("http://localhost:8080")
            manager.current_engine_client = mock_client

            client = manager.get_current_client()
            self.assertIs(client, mock_client)

        asyncio.run(run_test())

    def test_is_same_variant_no_current(self):
        """Test _is_same_variant when no current variant exists."""
        manager = EngineManager()

        variant_config = {
            "binary": "/usr/bin/llama-server",
            "connect": "http://localhost:8080",
            "context": 2048,
        }

        is_same = manager._is_same_variant("test-model", variant_config)
        self.assertFalse(is_same)

    def test_is_same_variant_different_model(self):
        """Test _is_same_variant with different model name."""
        manager = EngineManager()

        manager.current_model_name = "model-a"
        manager.current_variant_config = {
            "binary": "/usr/bin/llama-server",
            "connect": "http://localhost:8080",
            "context": 2048,
        }

        variant_config = {
            "binary": "/usr/bin/llama-server",
            "connect": "http://localhost:8080",
            "context": 2048,
        }

        is_same = manager._is_same_variant("model-b", variant_config)
        self.assertFalse(is_same)

    def test_is_same_variant_different_binary(self):
        """Test _is_same_variant with different binary."""
        manager = EngineManager()

        manager.current_model_name = "test-model"
        manager.current_variant_config = {
            "binary": "/usr/bin/llama-server",
            "connect": "http://localhost:8080",
            "context": 2048,
        }

        variant_config = {
            "binary": "/usr/bin/llama-server-new",
            "connect": "http://localhost:8080",
            "context": 2048,
        }

        is_same = manager._is_same_variant("test-model", variant_config)
        self.assertFalse(is_same)

    def test_is_same_variant_different_connect(self):
        """Test _is_same_variant with different connect URL."""
        manager = EngineManager()

        manager.current_model_name = "test-model"
        manager.current_variant_config = {
            "binary": "/usr/bin/llama-server",
            "connect": "http://localhost:8080",
            "context": 2048,
        }

        variant_config = {
            "binary": "/usr/bin/llama-server",
            "connect": "http://localhost:8081",
            "context": 2048,
        }

        is_same = manager._is_same_variant("test-model", variant_config)
        self.assertFalse(is_same)

    def test_is_same_variant_different_context(self):
        """Test _is_same_variant with different context size."""
        manager = EngineManager()

        manager.current_model_name = "test-model"
        manager.current_variant_config = {
            "binary": "/usr/bin/llama-server",
            "connect": "http://localhost:8080",
            "context": 2048,
        }

        variant_config = {
            "binary": "/usr/bin/llama-server",
            "connect": "http://localhost:8080",
            "context": 4096,
        }

        is_same = manager._is_same_variant("test-model", variant_config)
        self.assertFalse(is_same)

    def test_is_same_variant_identical(self):
        """Test _is_same_variant with identical configuration."""
        manager = EngineManager()

        manager.current_model_name = "test-model"
        manager.current_variant_config = {
            "binary": "/usr/bin/llama-server",
            "connect": "http://localhost:8080",
            "context": 2048,
        }

        variant_config = {
            "binary": "/usr/bin/llama-server",
            "connect": "http://localhost:8080",
            "context": 2048,
        }

        is_same = manager._is_same_variant("test-model", variant_config)
        self.assertTrue(is_same)

    def test_stop_current_engine_no_engine(self):
        """Test stopping current engine when none is running."""

        async def run_test():
            manager = EngineManager()

            # Should not raise an error
            await manager._stop_current_engine()

            self.assertIsNone(manager.current_engine_process)

        asyncio.run(run_test())

    def test_stop_current_engine_with_running_engine(self):
        """Test stopping a running engine."""

        async def run_test():
            manager = EngineManager()

            # Create mock process
            mock_process = AsyncMock(spec=EngineProcess)
            mock_process.stop = AsyncMock()

            manager.current_engine_process = mock_process
            manager.current_engine_client = MockEngineClient("http://localhost:8080")
            manager.current_variant_config = {"binary": "test"}
            manager.current_model_name = "test-model"

            await manager._stop_current_engine()

            # Verify stop was called
            mock_process.stop.assert_called_once_with(timeout=15.0)

            # Verify state was cleared
            self.assertIsNone(manager.current_engine_process)
            self.assertIsNone(manager.current_engine_client)
            self.assertIsNone(manager.current_variant_config)
            self.assertIsNone(manager.current_model_name)

        asyncio.run(run_test())

    def test_stop_current_engine_with_error(self):
        """Test stopping engine handles errors gracefully."""

        async def run_test():
            manager = EngineManager()

            # Create mock process that raises error
            mock_process = AsyncMock(spec=EngineProcess)
            mock_process.stop = AsyncMock(side_effect=Exception("Stop failed"))

            manager.current_engine_process = mock_process

            # Should not raise, but should log error
            await manager._stop_current_engine()

            # State should still be cleared
            self.assertIsNone(manager.current_engine_process)

        asyncio.run(run_test())

    def test_start_new_engine_llama_cpp(self):
        """Test starting a new llama.cpp engine."""

        async def run_test():
            manager = EngineManager()

            model_name = "test-model"
            variant_config = {
                "binary": "/usr/bin/llama-server",
                "args": ["-c", "2048"],
                "connect": "http://localhost:8080",
            }
            engine_type = "llama.cpp"

            # Mock EngineProcess
            mock_process = AsyncMock(spec=EngineProcess)
            mock_process.start = AsyncMock()

            # Mock health check to return True immediately
            with patch(
                "engine.engine_manager.EngineProcess", return_value=mock_process
            ):
                with patch("engine.engine_manager.LlamaCppEngine") as mock_engine_class:
                    mock_client = MockEngineClient("http://localhost:8080")
                    mock_client.check_health = AsyncMock(return_value=True)
                    mock_engine_class.return_value = mock_client

                    await manager._start_new_engine(
                        model_name, variant_config, engine_type
                    )

                    # Verify process was started
                    mock_process.start.assert_called_once()

                    # Verify state was set
                    self.assertEqual(manager.current_model_name, model_name)
                    self.assertEqual(manager.current_variant_config, variant_config)
                    self.assertIsNotNone(manager.current_engine_client)
                    self.assertIsNotNone(manager.current_engine_process)

        asyncio.run(run_test())

    def test_start_new_engine_unknown_type(self):
        """Test starting engine with unknown type raises error."""

        async def run_test():
            manager = EngineManager()

            variant_config = {
                "binary": "/usr/bin/unknown",
                "args": [],
                "connect": "http://localhost:8080",
            }

            with self.assertRaises(ValueError) as context:
                await manager._start_new_engine(
                    "test-model", variant_config, "unknown-engine"
                )

            self.assertIn("Unknown engine type", str(context.exception))

        asyncio.run(run_test())

    def test_start_new_engine_missing_binary(self):
        """Test starting engine with missing binary field."""

        async def run_test():
            manager = EngineManager()

            variant_config = {
                "connect": "http://localhost:8080",
            }

            with self.assertRaises(ValueError) as context:
                await manager._start_new_engine(
                    "test-model", variant_config, "llama.cpp"
                )

            self.assertIn("missing 'binary'", str(context.exception))

        asyncio.run(run_test())

    def test_start_new_engine_missing_connect(self):
        """Test starting engine with missing connect field."""

        async def run_test():
            manager = EngineManager()

            variant_config = {
                "binary": "/usr/bin/llama-server",
            }

            with self.assertRaises(ValueError) as context:
                await manager._start_new_engine(
                    "test-model", variant_config, "llama.cpp"
                )

            self.assertIn("missing 'connect'", str(context.exception))

        asyncio.run(run_test())

    def test_start_new_engine_process_start_fails(self):
        """Test handling of process start failure."""

        async def run_test():
            manager = EngineManager()

            variant_config = {
                "binary": "/usr/bin/llama-server",
                "args": [],
                "connect": "http://localhost:8080",
            }

            mock_process = AsyncMock(spec=EngineProcess)
            mock_process.start = AsyncMock(
                side_effect=FileNotFoundError("Binary not found")
            )

            with patch(
                "engine.engine_manager.EngineProcess", return_value=mock_process
            ):
                with self.assertRaises(FileNotFoundError):
                    await manager._start_new_engine(
                        "test-model", variant_config, "llama.cpp"
                    )

        asyncio.run(run_test())

    def test_start_new_engine_health_check_timeout(self):
        """Test handling of engine not becoming ready."""

        async def run_test():
            manager = EngineManager()

            variant_config = {
                "binary": "/usr/bin/llama-server",
                "args": [],
                "connect": "http://localhost:8080",
            }

            mock_process = AsyncMock(spec=EngineProcess)
            mock_process.start = AsyncMock()
            mock_process.stop = AsyncMock()

            with patch(
                "engine.engine_manager.EngineProcess", return_value=mock_process
            ):
                with patch("engine.engine_manager.LlamaCppEngine") as mock_engine_class:
                    mock_client = MockEngineClient("http://localhost:8080")
                    # Health check always fails
                    mock_client.check_health = AsyncMock(return_value=False)
                    mock_engine_class.return_value = mock_client

                    # Reduce timeout for faster test
                    with patch.object(manager, "_wait_for_engine_ready") as mock_wait:
                        mock_wait.side_effect = TimeoutError("Engine not ready")

                        with self.assertRaises(TimeoutError):
                            await manager._start_new_engine(
                                "test-model", variant_config, "llama.cpp"
                            )

                        # Process should be stopped
                        mock_process.stop.assert_called_once()

        asyncio.run(run_test())

    def test_wait_for_engine_ready_success(self):
        """Test waiting for engine to become ready - success case."""

        async def run_test():
            manager = EngineManager()

            mock_client = MockEngineClient("http://localhost:8080")
            mock_client.check_health = AsyncMock(return_value=True)

            result = await manager._wait_for_engine_ready(mock_client, timeout=5.0)

            self.assertTrue(result)
            mock_client.check_health.assert_called_once()

        asyncio.run(run_test())

    def test_wait_for_engine_ready_eventual_success(self):
        """Test waiting for engine that becomes ready after a few attempts."""

        async def run_test():
            manager = EngineManager()

            mock_client = MockEngineClient("http://localhost:8080")

            # First two calls fail, third succeeds
            mock_client.check_health = AsyncMock(side_effect=[False, False, True])

            result = await manager._wait_for_engine_ready(mock_client, timeout=10.0)

            self.assertTrue(result)
            self.assertEqual(mock_client.check_health.call_count, 3)

        asyncio.run(run_test())

    def test_wait_for_engine_ready_timeout(self):
        """Test waiting for engine times out."""

        async def run_test():
            manager = EngineManager()

            mock_client = MockEngineClient("http://localhost:8080")
            # Always fails
            mock_client.check_health = AsyncMock(return_value=False)

            with self.assertRaises(TimeoutError) as context:
                await manager._wait_for_engine_ready(mock_client, timeout=2.0)

            self.assertIn("did not become ready", str(context.exception))

        asyncio.run(run_test())

    def test_ensure_engine_same_variant_healthy(self):
        """Test ensure_engine when same variant is already running and healthy."""

        async def run_test():
            manager = EngineManager()

            model_name = "test-model"
            variant_config = {
                "binary": "/usr/bin/llama-server",
                "connect": "http://localhost:8080",
                "context": 2048,
            }

            # Set up current state
            mock_client = MockEngineClient("http://localhost:8080")
            mock_client.check_health = AsyncMock(return_value=True)

            manager.current_model_name = model_name
            manager.current_variant_config = variant_config
            manager.current_engine_client = mock_client

            # Should return same client without restarting
            client = await manager.ensure_engine(
                model_name, variant_config, "llama.cpp"
            )

            self.assertIs(client, mock_client)
            mock_client.check_health.assert_called_once()

        asyncio.run(run_test())

    def test_ensure_engine_same_variant_unhealthy(self):
        """Test ensure_engine when same variant is unhealthy - should restart."""

        async def run_test():
            manager = EngineManager()

            model_name = "test-model"
            variant_config = {
                "binary": "/usr/bin/llama-server",
                "args": [],
                "connect": "http://localhost:8080",
                "context": 2048,
            }

            # Set up current state with unhealthy client
            old_client = MockEngineClient("http://localhost:8080")
            old_client.check_health = AsyncMock(return_value=False)

            mock_process = AsyncMock(spec=EngineProcess)
            mock_process.stop = AsyncMock()

            manager.current_model_name = model_name
            manager.current_variant_config = variant_config
            manager.current_engine_client = old_client
            manager.current_engine_process = mock_process

            # Mock starting new engine
            with patch.object(
                manager, "_stop_current_engine", new_callable=AsyncMock
            ) as mock_stop:
                with patch.object(
                    manager, "_start_new_engine", new_callable=AsyncMock
                ) as mock_start:
                    new_client = MockEngineClient("http://localhost:8080")
                    manager.current_engine_client = new_client

                    client = await manager.ensure_engine(
                        model_name, variant_config, "llama.cpp"
                    )

                    # Should have stopped and started
                    mock_stop.assert_called_once()
                    mock_start.assert_called_once_with(
                        model_name, variant_config, "llama.cpp"
                    )

        asyncio.run(run_test())

    def test_ensure_engine_different_variant(self):
        """Test ensure_engine when different variant is needed."""

        async def run_test():
            manager = EngineManager()

            old_variant = {
                "binary": "/usr/bin/llama-server",
                "args": [],
                "connect": "http://localhost:8080",
                "context": 2048,
            }

            new_variant = {
                "binary": "/usr/bin/llama-server",
                "args": [],
                "connect": "http://localhost:8080",
                "context": 4096,
            }

            # Set up current state
            manager.current_model_name = "test-model"
            manager.current_variant_config = old_variant
            manager.current_engine_client = MockEngineClient("http://localhost:8080")

            # Mock engine switching
            with patch.object(
                manager, "_stop_current_engine", new_callable=AsyncMock
            ) as mock_stop:
                with patch.object(
                    manager, "_start_new_engine", new_callable=AsyncMock
                ) as mock_start:
                    new_client = MockEngineClient("http://localhost:8080")
                    manager.current_engine_client = new_client

                    client = await manager.ensure_engine(
                        "test-model", new_variant, "llama.cpp"
                    )

                    # Should have stopped old and started new
                    mock_stop.assert_called_once()
                    mock_start.assert_called_once_with(
                        "test-model", new_variant, "llama.cpp"
                    )

        asyncio.run(run_test())

    def test_ensure_engine_concurrent_requests(self):
        """Test that concurrent ensure_engine calls are properly serialized."""

        async def run_test():
            manager = EngineManager()

            variant_config = {
                "binary": "/usr/bin/llama-server",
                "args": [],
                "connect": "http://localhost:8080",
                "context": 2048,
            }

            call_order = []

            async def mock_start(model_name, variant_config, engine_type):
                call_order.append("start_begin")
                await asyncio.sleep(0.1)
                call_order.append("start_end")
                manager.current_engine_client = MockEngineClient(
                    "http://localhost:8080"
                )
                manager.current_model_name = model_name
                manager.current_variant_config = variant_config

            with patch.object(manager, "_start_new_engine", side_effect=mock_start):
                with patch.object(
                    manager, "_stop_current_engine", new_callable=AsyncMock
                ):
                    # Start two concurrent ensure_engine calls
                    task1 = asyncio.create_task(
                        manager.ensure_engine("test-model", variant_config, "llama.cpp")
                    )
                    task2 = asyncio.create_task(
                        manager.ensure_engine("test-model", variant_config, "llama.cpp")
                    )

                    await asyncio.gather(task1, task2)

                    # Calls should be serialized - both starts should complete sequentially
                    self.assertIn("start_begin", call_order)
                    self.assertIn("start_end", call_order)

        asyncio.run(run_test())

    def test_shutdown(self):
        """Test shutdown stops current engine."""

        async def run_test():
            manager = EngineManager()

            # Set up current state
            mock_process = AsyncMock(spec=EngineProcess)
            mock_process.stop = AsyncMock()

            manager.current_engine_process = mock_process
            manager.current_engine_client = MockEngineClient("http://localhost:8080")
            manager.current_model_name = "test-model"

            await manager.shutdown()

            # Should have stopped engine
            mock_process.stop.assert_called_once()

            # State should be cleared
            self.assertIsNone(manager.current_engine_process)

        asyncio.run(run_test())

    def test_shutdown_with_no_engine(self):
        """Test shutdown when no engine is running."""

        async def run_test():
            manager = EngineManager()

            # Should not raise
            await manager.shutdown()

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
