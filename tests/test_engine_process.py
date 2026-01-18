import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from engine.engine_process import EngineProcess


class TestEngineProcess(unittest.IsolatedAsyncioTestCase):
    """Test the EngineProcess class for subprocess management."""

    def setUp(self):
        """Set up test fixtures."""
        self.binary_path = "/path/to/engine"
        self.args = ["--arg1", "value1", "--arg2", "value2"]
        self.work_dir = "/tmp/test"
        self.engine = EngineProcess(
            binary_path=self.binary_path,
            args=self.args,
            work_dir=self.work_dir,
        )

    async def test_start_success(self):
        """Test successful process start."""
        # Create a mock process
        mock_process = MagicMock(spec=asyncio.subprocess.Process)
        mock_process.pid = 12345
        mock_process.returncode = None  # Process is running
        mock_process.stdout = AsyncMock(spec=asyncio.StreamReader)
        mock_process.stderr = AsyncMock(spec=asyncio.StreamReader)

        # Mock stdout/stderr readline to simulate process output
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                b"Engine started successfully\n",
                b"",
                b"",  # Empty bytes indicate EOF
            ]
        )
        mock_process.stderr.readline = AsyncMock(
            side_effect=[
                b"Warning: Some warning\n",
                b"",
            ]
        )

        with patch(
            "asyncio.create_subprocess_exec", AsyncMock(return_value=mock_process)
        ):
            await self.engine.start()

            # Verify process was started
            self.assertTrue(self.engine.is_running())
            self.assertEqual(self.engine.get_pid(), 12345)
            self.assertEqual(self.engine.get_status(), "running")
            self.assertIsNotNone(self.engine.get_uptime())

            # Verify subprocess was created with correct arguments
            asyncio.create_subprocess_exec.assert_called_once_with(
                self.binary_path,
                *self.args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.work_dir,
            )

    async def test_start_already_running(self):
        """Test that starting an already running process raises error."""
        # Mock a running process
        mock_process = MagicMock(spec=asyncio.subprocess.Process)
        mock_process.pid = 12345
        mock_process.returncode = None
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()

        with patch(
            "asyncio.create_subprocess_exec", AsyncMock(return_value=mock_process)
        ):
            await self.engine.start()

            # Try to start again
            with self.assertRaises(RuntimeError) as cm:
                await self.engine.start()
            self.assertIn("already running", str(cm.exception))

    async def test_start_binary_not_found(self):
        """Test starting with non-existent binary raises error."""
        with patch(
            "asyncio.create_subprocess_exec",
            AsyncMock(side_effect=FileNotFoundError("Binary not found")),
        ):
            with self.assertRaises(FileNotFoundError):
                await self.engine.start()

            # Status should be failed
            self.assertEqual(self.engine.get_status(), "failed")

    async def test_stop_graceful(self):
        """Test graceful process stop."""
        # Start a mock process
        mock_process = MagicMock(spec=asyncio.subprocess.Process)
        mock_process.pid = 12345
        mock_process.returncode = None
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.wait = AsyncMock(return_value=0)  # Exit code 0

        with patch(
            "asyncio.create_subprocess_exec", AsyncMock(return_value=mock_process)
        ):
            await self.engine.start()
            await self.engine.stop()

            # Verify terminate was called
            mock_process.terminate.assert_called_once()
            mock_process.wait.assert_called_once()

            # Verify process is stopped
            self.assertFalse(self.engine.is_running())
            self.assertEqual(self.engine.get_status(), "stopped")

    async def test_stop_forceful(self):
        """Test forceful process stop after timeout."""
        # Start a mock process
        mock_process = MagicMock(spec=asyncio.subprocess.Process)
        mock_process.pid = 12345
        mock_process.returncode = None
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()

        # Simulate timeout on graceful shutdown
        mock_process.wait = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch(
            "asyncio.create_subprocess_exec", AsyncMock(return_value=mock_process)
        ):
            await self.engine.start()
            await self.engine.stop(timeout=0.1)  # Short timeout for test

            # Verify both terminate and kill were called
            mock_process.terminate.assert_called_once()
            mock_process.kill.assert_called_once()
            mock_process.wait.assert_called()

            # Verify process is stopped
            self.assertFalse(self.engine.is_running())

    async def test_stop_not_running(self):
        """Test stopping a process that isn't running."""
        # Process was never started
        self.assertFalse(self.engine.is_running())

        # Should not raise an error
        await self.engine.stop()

        # Status should remain initialized
        self.assertEqual(self.engine.get_status(), "initialized")

    def test_is_running(self):
        """Test the is_running method in various states."""
        # Initially not running
        self.assertFalse(self.engine.is_running())

        # Mock a running process
        mock_process = MagicMock(spec=asyncio.subprocess.Process)
        mock_process.returncode = None
        self.engine._process = mock_process
        self.assertTrue(self.engine.is_running())

        # Mock a terminated process
        mock_process.returncode = 0
        self.assertFalse(self.engine.is_running())

        # No process object
        self.engine._process = None
        self.assertFalse(self.engine.is_running())

    def test_get_pid(self):
        """Test getting the process ID."""
        # No process
        self.assertIsNone(self.engine.get_pid())

        # With process
        mock_process = MagicMock(spec=asyncio.subprocess.Process)
        mock_process.pid = 12345
        self.engine._process = mock_process
        self.assertEqual(self.engine.get_pid(), 12345)

    def test_get_status(self):
        """Test status detection in various states."""
        # Initial state
        self.assertEqual(self.engine.get_status(), "initialized")

        # Running state
        mock_process = MagicMock(spec=asyncio.subprocess.Process)
        mock_process.returncode = None
        self.engine._process = mock_process
        self.engine._status = "running"
        self.assertEqual(self.engine.get_status(), "running")

        # Process crashed (returncode set, not stopped by us)
        mock_process.returncode = 1
        self.engine._stopped_by_us = False
        self.assertEqual(self.engine.get_status(), "crashed")

        # Process stopped by us
        mock_process.returncode = 0
        self.engine._stopped_by_us = True
        self.assertEqual(self.engine.get_status(), "stopped")

        # Failed state
        self.engine._status = "failed"
        self.assertEqual(self.engine.get_status(), "failed")

    def test_get_uptime(self):
        """Test uptime calculation."""
        # No start time
        self.assertIsNone(self.engine.get_uptime())

        # Set start time and running process
        start_time = time.time() - 5.0  # 5 seconds ago
        self.engine._start_time = start_time

        mock_process = MagicMock(spec=asyncio.subprocess.Process)
        mock_process.returncode = None
        self.engine._process = mock_process

        uptime = self.engine.get_uptime()
        self.assertIsNotNone(uptime)
        self.assertGreaterEqual(uptime, 4.9)
        self.assertLessEqual(uptime, 5.1)

        # Process not running (should return None even with start_time)
        mock_process.returncode = 0
        self.assertIsNone(self.engine.get_uptime())

    async def test_stdout_capture(self):
        """Test stdout log capture."""
        mock_process = MagicMock(spec=asyncio.subprocess.Process)
        mock_process.pid = 12345
        mock_process.returncode = None
        mock_process.stdout = AsyncMock(spec=asyncio.StreamReader)
        mock_process.stderr = AsyncMock(spec=asyncio.StreamReader)

        # Simulate stdout lines
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                b"Line 1\n",
                b"Line 2\n",
                b"",  # EOF
            ]
        )
        mock_process.stderr.readline = AsyncMock(return_value=b"")

        with patch(
            "asyncio.create_subprocess_exec", AsyncMock(return_value=mock_process)
        ):
            await self.engine.start()

            # Give the reader tasks time to process
            await asyncio.sleep(0.01)

            # Cancel reader tasks to stop them
            if self.engine._stdout_task:
                self.engine._stdout_task.cancel()
                try:
                    await self.engine._stdout_task
                except asyncio.CancelledError:
                    pass

    async def test_stderr_capture(self):
        """Test stderr log capture."""
        mock_process = MagicMock(spec=asyncio.subprocess.Process)
        mock_process.pid = 12345
        mock_process.returncode = None
        mock_process.stdout = AsyncMock(spec=asyncio.StreamReader)
        mock_process.stderr = AsyncMock(spec=asyncio.StreamReader)

        # Simulate stderr lines
        mock_process.stdout.readline = AsyncMock(return_value=b"")
        mock_process.stderr.readline = AsyncMock(
            side_effect=[
                b"Error: Something went wrong\n",
                b"Warning: This is a warning\n",
                b"",  # EOF
            ]
        )

        with patch(
            "asyncio.create_subprocess_exec", AsyncMock(return_value=mock_process)
        ):
            await self.engine.start()

            # Give the reader tasks time to process
            await asyncio.sleep(0.01)

            # Cancel reader tasks to stop them
            if self.engine._stderr_task:
                self.engine._stderr_task.cancel()
                try:
                    await self.engine._stderr_task
                except asyncio.CancelledError:
                    pass

    async def test_process_crash_detection(self):
        """Test that process crashes are detected."""
        mock_process = MagicMock(spec=asyncio.subprocess.Process)
        mock_process.pid = 12345
        mock_process.returncode = None  # Initially running
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()

        with patch(
            "asyncio.create_subprocess_exec", AsyncMock(return_value=mock_process)
        ):
            await self.engine.start()

            # Simulate process crash
            mock_process.returncode = 1
            self.engine._stopped_by_us = False

            # Status should update to "crashed"
            self.assertEqual(self.engine.get_status(), "crashed")

    def test_string_representation(self):
        """Test the string representation of EngineProcess."""
        # Test with no process
        result = str(self.engine)
        self.assertIn("EngineProcess", result)
        self.assertIn("pid=N/A", result)
        self.assertIn("status=initialized", result)

        # Test with running process
        mock_process = MagicMock(spec=asyncio.subprocess.Process)
        mock_process.pid = 12345
        mock_process.returncode = None
        self.engine._process = mock_process
        self.engine._status = "running"
        self.engine._start_time = time.time() - 10.0

        result = str(self.engine)
        self.assertIn("pid=12345", result)
        self.assertIn("status=running", result)
        self.assertIn("uptime=10.0s", result)

    async def test_concurrent_stop_protection(self):
        """Test that stop can be called multiple times safely."""
        mock_process = MagicMock(spec=asyncio.subprocess.Process)
        mock_process.pid = 12345
        mock_process.returncode = None
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.wait = AsyncMock(return_value=0)

        with patch(
            "asyncio.create_subprocess_exec", AsyncMock(return_value=mock_process)
        ):
            await self.engine.start()

            # Call stop multiple times concurrently
            await asyncio.gather(
                self.engine.stop(),
                self.engine.stop(),
                self.engine.stop(),
            )

            # Should only stop once
            mock_process.terminate.assert_called_once()

    async def test_reader_task_cancellation_on_stop(self):
        """Test that reader tasks are cancelled when process stops."""
        mock_process = MagicMock(spec=asyncio.subprocess.Process)
        mock_process.pid = 12345
        mock_process.returncode = None
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.wait = AsyncMock(return_value=0)

        # Mock readline to hang until cancelled
        readline_event = asyncio.Event()

        async def hanging_readline():
            await readline_event.wait()
            return b""

        mock_process.stdout.readline = AsyncMock(side_effect=hanging_readline)
        mock_process.stderr.readline = AsyncMock(side_effect=hanging_readline)

        with patch(
            "asyncio.create_subprocess_exec", AsyncMock(return_value=mock_process)
        ):
            await self.engine.start()

            # Verify reader tasks were created
            self.assertIsNotNone(self.engine._stdout_task)
            self.assertIsNotNone(self.engine._stderr_task)

            # Stop the process (should cancel reader tasks)
            stop_task = asyncio.create_task(self.engine.stop())

            # Release the readline hangs
            readline_event.set()

            await stop_task

            # Reader tasks should be done (cancelled)
            self.assertTrue(self.engine._stdout_task.done())
            self.assertTrue(self.engine._stderr_task.done())


if __name__ == "__main__":
    unittest.main()
