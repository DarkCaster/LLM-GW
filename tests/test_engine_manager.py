import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, call

from engine.engine_manager import EngineManager
from engine.engine_process import EngineProcess
from engine.llamacpp_engine import LlamaCppEngine


class TestEngineManager(unittest.IsolatedAsyncioTestCase):
    """Test the EngineManager coordination logic."""

    def setUp(self):
        """Set up test fixtures and reset singleton instance."""
        # Reset the singleton instance before each test
        EngineManager._instance = None
        self.manager = EngineManager()

        # Test configuration
        self.model_name = "test-model"
        self.variant_config = {
            "binary": "/path/to/engine",
            "args": ["--arg1", "value1"],
            "connect": "http://127.0.0.1:8080",
            "context": 32000,
            "tokenize": True,
        }
        self.engine_type = "llama.cpp"

    def tearDown(self):
        """Clean up after tests."""
        if EngineManager._instance:
            EngineManager._instance = None

    def test_singleton_pattern(self):
        """Test that EngineManager follows singleton pattern."""
        manager1 = EngineManager()
        manager2 = EngineManager()

        self.assertIs(manager1, manager2)
        self.assertIs(EngineManager._instance, manager1)

    async def test_ensure_engine_new_engine(self):
        """Test starting a new engine when none is running."""
        # Mock the engine client and process
        mock_engine_client = AsyncMock(spec=LlamaCppEngine)
        mock_engine_client.check_health = AsyncMock(return_value=True)

        mock_engine_process = AsyncMock(spec=EngineProcess)
        mock_engine_process.start = AsyncMock()
        mock_engine_process.get_pid = MagicMock(return_value=12345)
        mock_engine_process.is_running = MagicMock(return_value=True)

        with patch(
            "engine.engine_manager.LlamaCppEngine", return_value=mock_engine_client
        ):
            with patch(
                "engine.engine_manager.EngineProcess", return_value=mock_engine_process
            ):
                client = await self.manager.ensure_engine(
                    model_name=self.model_name,
                    variant_config=self.variant_config,
                    engine_type=self.engine_type,
                )

                # Verify engine process was created with correct arguments
                EngineProcess.assert_called_once_with(
                    binary_path=self.variant_config["binary"],
                    args=self.variant_config["args"],
                    logger=self.manager._logger,
                )

                # Verify process was started
                mock_engine_process.start.assert_called_once()

                # Verify engine client was created
                LlamaCppEngine.assert_called_once_with(
                    base_url=self.variant_config["connect"]
                )

                # Verify health check was performed
                mock_engine_client.check_health.assert_called_once()

                # Verify client was returned
                self.assertEqual(client, mock_engine_client)

                # Verify state was updated
                self.assertEqual(self.manager.current_model_name, self.model_name)
                self.assertEqual(
                    self.manager.current_variant_config, self.variant_config
                )

    async def test_ensure_engine_same_engine_already_running(self):
        """Test returning existing client when same engine is already running."""
        # Set up current engine state
        self.manager.current_model_name = self.model_name
        self.manager.current_variant_config = self.variant_config.copy()
        self.manager.current_engine_client = AsyncMock(spec=LlamaCppEngine)
        self.manager.current_engine_client.check_health = AsyncMock(return_value=True)

        client = await self.manager.ensure_engine(
            model_name=self.model_name,
            variant_config=self.variant_config,
            engine_type=self.engine_type,
        )

        # Should return existing client without starting new engine
        self.assertEqual(client, self.manager.current_engine_client)
        # Health check should have been called
        self.manager.current_engine_client.check_health.assert_called_once()

    async def test_ensure_engine_different_model_switches(self):
        """Test engine switching when different model is requested."""
        # Set up current engine state with different model
        self.manager.current_model_name = "different-model"
        self.manager.current_variant_config = {
            "binary": "/other/path",
            "args": [],
            "connect": "http://other:8080",
        }
        self.manager.current_engine_process = AsyncMock(spec=EngineProcess)
        self.manager.current_engine_process.stop = AsyncMock()

        mock_new_engine_client = AsyncMock(spec=LlamaCppEngine)
        mock_new_engine_client.check_health = AsyncMock(return_value=True)

        mock_new_engine_process = AsyncMock(spec=EngineProcess)
        mock_new_engine_process.start = AsyncMock()
        mock_new_engine_process.get_pid = MagicMock(return_value=67890)

        with patch(
            "engine.engine_manager.LlamaCppEngine", return_value=mock_new_engine_client
        ):
            with patch(
                "engine.engine_manager.EngineProcess",
                return_value=mock_new_engine_process,
            ):
                client = await self.manager.ensure_engine(
                    model_name=self.model_name,
                    variant_config=self.variant_config,
                    engine_type=self.engine_type,
                )

                # Verify old engine was stopped
                self.manager.current_engine_process.stop.assert_called_once()

                # Verify new engine was started
                mock_new_engine_process.start.assert_called_once()

                # Verify state was updated
                self.assertEqual(self.manager.current_model_name, self.model_name)
                self.assertEqual(
                    self.manager.current_variant_config, self.variant_config
                )

    async def test_ensure_engine_different_context_switches(self):
        """Test engine switching when different context size is requested."""
        # Set up current engine state with different context
        self.manager.current_model_name = self.model_name
        self.manager.current_variant_config = self.variant_config.copy()
        self.manager.current_variant_config["context"] = 16000  # Different context size

        self.manager.current_engine_process = AsyncMock(spec=EngineProcess)
        self.manager.current_engine_process.stop = AsyncMock()

        mock_new_engine_client = AsyncMock(spec=LlamaCppEngine)
        mock_new_engine_client.check_health = AsyncMock(return_value=True)

        mock_new_engine_process = AsyncMock(spec=EngineProcess)
        mock_new_engine_process.start = AsyncMock()

        with patch(
            "engine.engine_manager.LlamaCppEngine", return_value=mock_new_engine_client
        ):
            with patch(
                "engine.engine_manager.EngineProcess",
                return_value=mock_new_engine_process,
            ):
                await self.manager.ensure_engine(
                    model_name=self.model_name,
                    variant_config=self.variant_config,  # Original config with 32000 context
                    engine_type=self.engine_type,
                )

                # Verify engine was switched due to different context size
                self.manager.current_engine_process.stop.assert_called_once()
                mock_new_engine_process.start.assert_called_once()

    async def test_ensure_engine_health_check_fails(self):
        """Test engine switching when current engine fails health check."""
        # Set up current engine that fails health check
        self.manager.current_model_name = self.model_name
        self.manager.current_variant_config = self.variant_config.copy()
        self.manager.current_engine_client = AsyncMock(spec=LlamaCppEngine)
        self.manager.current_engine_client.check_health = AsyncMock(return_value=False)

        self.manager.current_engine_process = AsyncMock(spec=EngineProcess)
        self.manager.current_engine_process.stop = AsyncMock()

        mock_new_engine_client = AsyncMock(spec=LlamaCppEngine)
        mock_new_engine_client.check_health = AsyncMock(return_value=True)

        mock_new_engine_process = AsyncMock(spec=EngineProcess)
        mock_new_engine_process.start = AsyncMock()

        with patch(
            "engine.engine_manager.LlamaCppEngine", return_value=mock_new_engine_client
        ):
            with patch(
                "engine.engine_manager.EngineProcess",
                return_value=mock_new_engine_process,
            ):
                client = await self.manager.ensure_engine(
                    model_name=self.model_name,
                    variant_config=self.variant_config,
                    engine_type=self.engine_type,
                )

                # Verify old engine was stopped due to failed health check
                self.manager.current_engine_process.stop.assert_called_once()

                # Verify new engine was started
                mock_new_engine_process.start.assert_called_once()

                # Verify new client was returned
                self.assertEqual(client, mock_new_engine_client)

    async def test_ensure_engine_health_check_exception(self):
        """Test engine switching when health check raises exception."""
        # Set up current engine with health check that raises exception
        self.manager.current_model_name = self.model_name
        self.manager.current_variant_config = self.variant_config.copy()
        self.manager.current_engine_client = AsyncMock(spec=LlamaCppEngine)
        self.manager.current_engine_client.check_health = AsyncMock(
            side_effect=Exception("Health check failed")
        )

        self.manager.current_engine_process = AsyncMock(spec=EngineProcess)
        self.manager.current_engine_process.stop = AsyncMock()

        mock_new_engine_client = AsyncMock(spec=LlamaCppEngine)
        mock_new_engine_client.check_health = AsyncMock(return_value=True)

        mock_new_engine_process = AsyncMock(spec=EngineProcess)
        mock_new_engine_process.start = AsyncMock()

        with patch(
            "engine.engine_manager.LlamaCppEngine", return_value=mock_new_engine_client
        ):
            with patch(
                "engine.engine_manager.EngineProcess",
                return_value=mock_new_engine_process,
            ):
                client = await self.manager.ensure_engine(
                    model_name=self.model_name,
                    variant_config=self.variant_config,
                    engine_type=self.engine_type,
                )

                # Verify engine was switched due to health check exception
                self.manager.current_engine_process.stop.assert_called_once()
                mock_new_engine_process.start.assert_called_once()

    async def test_ensure_engine_process_start_fails(self):
        """Test error handling when engine process fails to start."""
        mock_engine_client = AsyncMock(spec=LlamaCppEngine)

        mock_engine_process = AsyncMock(spec=EngineProcess)
        mock_engine_process.start = AsyncMock(side_effect=Exception("Failed to start"))

        with patch(
            "engine.engine_manager.LlamaCppEngine", return_value=mock_engine_client
        ):
            with patch(
                "engine.engine_manager.EngineProcess", return_value=mock_engine_process
            ):
                with self.assertRaises(RuntimeError) as cm:
                    await self.manager.ensure_engine(
                        model_name=self.model_name,
                        variant_config=self.variant_config,
                        engine_type=self.engine_type,
                    )

                self.assertIn("Failed to start engine", str(cm.exception))

                # Verify state was cleaned up on failure
                self.assertIsNone(self.manager.current_engine_process)
                self.assertIsNone(self.manager.current_engine_client)

    async def test_ensure_engine_health_check_timeout(self):
        """Test engine fails to become ready within timeout."""
        mock_engine_client = AsyncMock(spec=LlamaCppEngine)
        mock_engine_client.check_health = AsyncMock(return_value=False)  # Always fails

        mock_engine_process = AsyncMock(spec=EngineProcess)
        mock_engine_process.start = AsyncMock()
        mock_engine_process.stop = AsyncMock()

        with patch(
            "engine.engine_manager.LlamaCppEngine", return_value=mock_engine_client
        ):
            with patch(
                "engine.engine_manager.EngineProcess", return_value=mock_engine_process
            ):
                with self.assertRaises(RuntimeError) as cm:
                    await self.manager.ensure_engine(
                        model_name=self.model_name,
                        variant_config=self.variant_config,
                        engine_type=self.engine_type,
                    )

                self.assertIn("Engine failed to become ready", str(cm.exception))

                # Verify process was cleaned up
                mock_engine_process.stop.assert_called_once()

    async def test_ensure_engine_missing_required_fields(self):
        """Test validation of required variant configuration fields."""
        invalid_config = {"binary": "/path/to/engine"}  # Missing args and connect

        with self.assertRaises(ValueError) as cm:
            await self.manager.ensure_engine(
                model_name=self.model_name,
                variant_config=invalid_config,
                engine_type=self.engine_type,
            )

        self.assertIn("Missing required field", str(cm.exception))

    async def test_ensure_engine_unsupported_engine_type(self):
        """Test error when unsupported engine type is requested."""
        with self.assertRaises(ValueError) as cm:
            await self.manager.ensure_engine(
                model_name=self.model_name,
                variant_config=self.variant_config,
                engine_type="unsupported-engine",
            )

        self.assertIn("Unsupported engine type", str(cm.exception))

    async def test_concurrent_ensure_engine_calls(self):
        """Test that concurrent calls to ensure_engine are properly serialized."""
        # Set up a slow engine start to test concurrency
        start_event = asyncio.Event()

        async def slow_health_check():
            await start_event.wait()
            return True

        mock_engine_client = AsyncMock(spec=LlamaCppEngine)
        mock_engine_client.check_health = AsyncMock(side_effect=slow_health_check)

        mock_engine_process = AsyncMock(spec=EngineProcess)
        mock_engine_process.start = AsyncMock()
        mock_engine_process.get_pid = MagicMock(return_value=12345)

        with patch(
            "engine.engine_manager.LlamaCppEngine", return_value=mock_engine_client
        ):
            with patch(
                "engine.engine_manager.EngineProcess", return_value=mock_engine_process
            ):
                # Start multiple concurrent ensure_engine calls
                tasks = [
                    asyncio.create_task(
                        self.manager.ensure_engine(
                            model_name=self.model_name,
                            variant_config=self.variant_config,
                            engine_type=self.engine_type,
                        )
                    )
                    for _ in range(3)
                ]

                # Allow the calls to proceed
                start_event.set()

                # Wait for all tasks to complete
                results = await asyncio.gather(*tasks)

                # All tasks should return the same client
                self.assertEqual(len(set(results)), 1)

                # Engine process should only be started once
                mock_engine_process.start.assert_called_once()

    async def test_shutdown_stops_current_engine(self):
        """Test that shutdown stops the currently running engine."""
        # Set up a running engine
        self.manager.current_engine_process = AsyncMock(spec=EngineProcess)
        self.manager.current_engine_process.stop = AsyncMock()
        self.manager.current_model_name = self.model_name

        await self.manager.shutdown()

        # Verify engine was stopped
        self.manager.current_engine_process.stop.assert_called_once()

        # Verify state was cleared
        self.assertIsNone(self.manager.current_engine_process)
        self.assertIsNone(self.manager.current_engine_client)
        self.assertIsNone(self.manager.current_model_name)
        self.assertIsNone(self.manager.current_variant_config)

    async def test_shutdown_no_engine_running(self):
        """Test shutdown when no engine is running."""
        # No engine is running initially
        self.assertIsNone(self.manager.current_engine_process)

        # Should not raise any errors
        await self.manager.shutdown()

    def test_get_current_client(self):
        """Test getting the current engine client."""
        # No client initially
        self.assertIsNone(self.manager.get_current_client())

        # Set a client
        mock_client = AsyncMock(spec=LlamaCppEngine)
        self.manager.current_engine_client = mock_client

        self.assertEqual(self.manager.get_current_client(), mock_client)

    def test_get_current_state(self):
        """Test getting current engine state for monitoring."""
        # State with no engine running
        state = self.manager.get_current_state()
        self.assertEqual(state["model_name"], None)
        self.assertEqual(state["engine_running"], False)
        self.assertEqual(state["variant_config"], None)

        # State with engine running
        mock_process = MagicMock(spec=EngineProcess)
        mock_process.get_pid = MagicMock(return_value=12345)
        mock_process.get_status = MagicMock(return_value="running")
        mock_process.get_uptime = MagicMock(return_value=10.5)

        self.manager.current_engine_process = mock_process
        self.manager.current_model_name = self.model_name
        self.manager.current_variant_config = self.variant_config

        state = self.manager.get_current_state()
        self.assertEqual(state["model_name"], self.model_name)
        self.assertEqual(state["engine_running"], True)
        self.assertEqual(state["variant_config"], self.variant_config)
        self.assertEqual(state["pid"], 12345)
        self.assertEqual(state["status"], "running")
        self.assertEqual(state["uptime"], 10.5)

    def test_is_same_engine_same_configuration(self):
        """Test engine comparison with same configuration."""
        self.manager.current_model_name = self.model_name
        self.manager.current_variant_config = self.variant_config.copy()

        result = self.manager._is_same_engine(self.model_name, self.variant_config)
        self.assertTrue(result)

    def test_is_same_engine_different_model(self):
        """Test engine comparison with different model."""
        self.manager.current_model_name = "different-model"
        self.manager.current_variant_config = self.variant_config.copy()

        result = self.manager._is_same_engine(self.model_name, self.variant_config)
        self.assertFalse(result)

    def test_is_same_engine_different_context(self):
        """Test engine comparison with different context size."""
        self.manager.current_model_name = self.model_name
        self.manager.current_variant_config = self.variant_config.copy()
        self.manager.current_variant_config["context"] = 16000

        result = self.manager._is_same_engine(self.model_name, self.variant_config)
        self.assertFalse(result)

    def test_is_same_engine_different_binary(self):
        """Test engine comparison with different binary."""
        self.manager.current_model_name = self.model_name
        self.manager.current_variant_config = self.variant_config.copy()
        self.manager.current_variant_config["binary"] = "/different/path"

        result = self.manager._is_same_engine(self.model_name, self.variant_config)
        self.assertFalse(result)

    def test_is_same_engine_no_current_engine(self):
        """Test engine comparison when no engine is running."""
        result = self.manager._is_same_engine(self.model_name, self.variant_config)
        self.assertFalse(result)

    async def test_stop_current_engine_no_engine(self):
        """Test stopping current engine when none is running."""
        # Should not raise any errors
        await self.manager._stop_current_engine()

    async def test_stop_current_engine_stop_fails(self):
        """Test error handling when stopping engine fails."""
        mock_process = AsyncMock(spec=EngineProcess)
        mock_process.stop = AsyncMock(side_effect=Exception("Stop failed"))
        self.manager.current_engine_process = mock_process
        self.manager.current_model_name = self.model_name

        # Should not raise exception (error is logged but not propagated)
        await self.manager._stop_current_engine()

        # State should still be cleared even if stop failed
        self.assertIsNone(self.manager.current_engine_process)
        self.assertIsNone(self.manager.current_model_name)

    def test_engine_type_mapping(self):
        """Test that engine type mapping is properly initialized."""
        # Check that llama.cpp is mapped to LlamaCppEngine
        self.assertIn("llama.cpp", self.manager._engine_types)
        self.assertEqual(self.manager._engine_types["llama.cpp"], LlamaCppEngine)


if __name__ == "__main__":
    unittest.main()
