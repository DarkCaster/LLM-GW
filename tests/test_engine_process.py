# tests/test_engine_process.py

import unittest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from engine.engine_process import EngineProcess


class TestEngineProcess(unittest.TestCase):
    """Test EngineProcess subprocess management."""

    def setUp(self):
        """Set up test fixtures."""
        self.binary_path = "/usr/bin/llama-server"
        self.args = ["-c", "2048", "-m", "model.gguf"]
        self.work_dir = "/tmp/test"

    def test_initialization(self):
        """Test EngineProcess initialization."""
        process = EngineProcess(self.binary_path, self.args, self.work_dir)

        self.assertEqual(process.binary_path, self.binary_path)
        self.assertEqual(process.args, self.args)
        self.assertEqual(process.work_dir, self.work_dir)
        self.assertIsNone(process._process)
        self.assertEqual(process.status, "stopped")
        self.assertIsNone(process.pid)

    def test_initialization_without_work_dir(self):
        """Test EngineProcess initialization without work directory."""
        process = EngineProcess(self.binary_path, self.args)

        self.assertEqual(process.binary_path, self.binary_path)
        self.assertEqual(process.args, self.args)
        self.assertIsNone(process.work_dir)

    def test_start_process(self):
        """Test starting a subprocess."""

        async def run_test():
            process = EngineProcess(self.binary_path, self.args, self.work_dir)

            mock_subprocess = AsyncMock()
            mock_subprocess.pid = 12345
            mock_subprocess.returncode = None
            mock_subprocess.stdout = AsyncMock()
            mock_subprocess.stdout.readline = AsyncMock(return_value=b"")
            mock_subprocess.stderr = AsyncMock()
            mock_subprocess.stderr.readline = AsyncMock(return_value=b"")

            with patch(
                "asyncio.create_subprocess_exec", return_value=mock_subprocess
            ) as mock_create:
                await process.start()

                # Verify subprocess was created with correct arguments
                mock_create.assert_called_once_with(
                    self.binary_path,
                    *self.args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.work_dir,
                )

                # Verify process state
                self.assertEqual(process.status, "running")
                self.assertEqual(process.pid, 12345)
                self.assertTrue(process.is_running)
                self.assertIsNotNone(process._start_time)

            # Clean up tasks
            if process._stdout_task:
                process._stdout_task.cancel()
            if process._stderr_task:
                process._stderr_task.cancel()

        asyncio.run(run_test())

    def test_start_process_already_running(self):
        """Test starting a process that is already running."""

        async def run_test():
            process = EngineProcess(self.binary_path, self.args)

            mock_subprocess = AsyncMock()
            mock_subprocess.pid = 12345
            mock_subprocess.returncode = None
            mock_subprocess.stdout = AsyncMock()
            mock_subprocess.stdout.readline = AsyncMock(return_value=b"")
            mock_subprocess.stderr = AsyncMock()
            mock_subprocess.stderr.readline = AsyncMock(return_value=b"")

            with patch(
                "asyncio.create_subprocess_exec", return_value=mock_subprocess
            ) as mock_create:
                await process.start()

                # Try to start again
                await process.start()

                # Should only be called once
                self.assertEqual(mock_create.call_count, 1)

            # Clean up tasks
            if process._stdout_task:
                process._stdout_task.cancel()
            if process._stderr_task:
                process._stderr_task.cancel()

        asyncio.run(run_test())

    def test_start_process_failure(self):
        """Test handling of subprocess start failure."""

        async def run_test():
            process = EngineProcess(self.binary_path, self.args)

            with patch(
                "asyncio.create_subprocess_exec",
                side_effect=FileNotFoundError("Binary not found"),
            ):
                with self.assertRaises(FileNotFoundError):
                    await process.start()

                # Verify process state
                self.assertEqual(process.status, "stopped")
                self.assertIsNone(process._process)

        asyncio.run(run_test())

    def test_stop_process_graceful(self):
        """Test graceful process termination."""

        async def run_test():
            process = EngineProcess(self.binary_path, self.args)

            mock_subprocess = AsyncMock()
            mock_subprocess.pid = 12345
            mock_subprocess.returncode = None
            mock_subprocess.stdout = AsyncMock()
            mock_subprocess.stdout.readline = AsyncMock(return_value=b"")
            mock_subprocess.stderr = AsyncMock()
            mock_subprocess.stderr.readline = AsyncMock(return_value=b"")
            mock_subprocess.terminate = Mock()
            mock_subprocess.wait = AsyncMock()

            with patch("asyncio.create_subprocess_exec", return_value=mock_subprocess):
                await process.start()

                # Simulate successful graceful termination
                async def wait_side_effect():
                    mock_subprocess.returncode = 0

                mock_subprocess.wait = AsyncMock(side_effect=wait_side_effect)

                await process.stop(timeout=5.0)

                # Verify terminate was called
                mock_subprocess.terminate.assert_called_once()

                # Verify status
                self.assertEqual(process.status, "stopped")
                self.assertFalse(process.is_running)

        asyncio.run(run_test())

    def test_stop_process_forceful(self):
        """Test forceful process termination after timeout."""

        async def run_test():
            process = EngineProcess(self.binary_path, self.args)

            mock_subprocess = AsyncMock()
            mock_subprocess.pid = 12345
            mock_subprocess.returncode = None
            mock_subprocess.stdout = AsyncMock()
            mock_subprocess.stdout.readline = AsyncMock(return_value=b"")
            mock_subprocess.stderr = AsyncMock()
            mock_subprocess.stderr.readline = AsyncMock(return_value=b"")
            mock_subprocess.terminate = Mock()
            mock_subprocess.kill = Mock()

            # First wait times out, second wait (after kill) succeeds
            async def wait_side_effect():
                if mock_subprocess.kill.call_count > 0:
                    mock_subprocess.returncode = -9
                else:
                    raise asyncio.TimeoutError()

            mock_subprocess.wait = AsyncMock(side_effect=wait_side_effect)

            with patch("asyncio.create_subprocess_exec", return_value=mock_subprocess):
                await process.start()

                await process.stop(timeout=0.1)

                # Verify both terminate and kill were called
                mock_subprocess.terminate.assert_called_once()
                mock_subprocess.kill.assert_called_once()

                # Verify status
                self.assertEqual(process.status, "stopped")

        asyncio.run(run_test())

    def test_stop_process_not_running(self):
        """Test stopping a process that is not running."""

        async def run_test():
            process = EngineProcess(self.binary_path, self.args)

            # Should not raise an error
            await process.stop()

            self.assertEqual(process.status, "stopped")

        asyncio.run(run_test())

    def test_stop_process_already_terminated(self):
        """Test stopping a process that already terminated."""

        async def run_test():
            process = EngineProcess(self.binary_path, self.args)

            mock_subprocess = AsyncMock()
            mock_subprocess.pid = 12345
            mock_subprocess.returncode = None
            mock_subprocess.stdout = AsyncMock()
            mock_subprocess.stdout.readline = AsyncMock(return_value=b"")
            mock_subprocess.stderr = AsyncMock()
            mock_subprocess.stderr.readline = AsyncMock(return_value=b"")
            mock_subprocess.terminate = Mock(side_effect=ProcessLookupError())

            with patch("asyncio.create_subprocess_exec", return_value=mock_subprocess):
                await process.start()

                await process.stop()

                # Should handle ProcessLookupError gracefully
                self.assertEqual(process.status, "stopped")

        asyncio.run(run_test())

    def test_is_running_property(self):
        """Test is_running property."""

        async def run_test():
            process = EngineProcess(self.binary_path, self.args)

            # Initially not running
            self.assertFalse(process.is_running)

            mock_subprocess = AsyncMock()
            mock_subprocess.pid = 12345
            mock_subprocess.returncode = None
            mock_subprocess.stdout = AsyncMock()
            mock_subprocess.stdout.readline = AsyncMock(return_value=b"")
            mock_subprocess.stderr = AsyncMock()
            mock_subprocess.stderr.readline = AsyncMock(return_value=b"")

            with patch("asyncio.create_subprocess_exec", return_value=mock_subprocess):
                await process.start()

                # Should be running
                self.assertTrue(process.is_running)

                # Simulate process exit
                mock_subprocess.returncode = 0

                # Should no longer be running
                self.assertFalse(process.is_running)

            # Clean up tasks
            if process._stdout_task:
                process._stdout_task.cancel()
            if process._stderr_task:
                process._stderr_task.cancel()

        asyncio.run(run_test())

    def test_status_property_transitions(self):
        """Test status property state transitions."""

        async def run_test():
            process = EngineProcess(self.binary_path, self.args)

            # Initially stopped
            self.assertEqual(process.status, "stopped")

            mock_subprocess = AsyncMock()
            mock_subprocess.pid = 12345
            mock_subprocess.returncode = None
            mock_subprocess.stdout = AsyncMock()
            mock_subprocess.stdout.readline = AsyncMock(return_value=b"")
            mock_subprocess.stderr = AsyncMock()
            mock_subprocess.stderr.readline = AsyncMock(return_value=b"")

            with patch("asyncio.create_subprocess_exec", return_value=mock_subprocess):
                await process.start()

                # Should be running
                self.assertEqual(process.status, "running")

                # Simulate clean exit
                mock_subprocess.returncode = 0
                status = process.status
                self.assertEqual(status, "stopped")

                # Simulate crashed exit
                process._status = "running"
                mock_subprocess.returncode = 1
                status = process.status
                self.assertEqual(status, "crashed")

            # Clean up tasks
            if process._stdout_task:
                process._stdout_task.cancel()
            if process._stderr_task:
                process._stderr_task.cancel()

        asyncio.run(run_test())

    def test_pid_property(self):
        """Test PID property."""

        async def run_test():
            process = EngineProcess(self.binary_path, self.args)

            # Initially no PID
            self.assertIsNone(process.pid)

            mock_subprocess = AsyncMock()
            mock_subprocess.pid = 12345
            mock_subprocess.returncode = None
            mock_subprocess.stdout = AsyncMock()
            mock_subprocess.stdout.readline = AsyncMock(return_value=b"")
            mock_subprocess.stderr = AsyncMock()
            mock_subprocess.stderr.readline = AsyncMock(return_value=b"")

            with patch("asyncio.create_subprocess_exec", return_value=mock_subprocess):
                await process.start()

                # Should have PID
                self.assertEqual(process.pid, 12345)

            # Clean up tasks
            if process._stdout_task:
                process._stdout_task.cancel()
            if process._stderr_task:
                process._stderr_task.cancel()

        asyncio.run(run_test())

    def test_stdout_capture(self):
        """Test stdout capture and logging."""

        async def run_test():
            process = EngineProcess(self.binary_path, self.args)

            # Create mock subprocess with stdout lines
            mock_subprocess = AsyncMock()
            mock_subprocess.pid = 12345
            mock_subprocess.returncode = None

            stdout_lines = [
                b"Starting server...\n",
                b"Server listening on port 8080\n",
                b"",  # EOF
            ]

            mock_subprocess.stdout = AsyncMock()
            mock_subprocess.stdout.readline = AsyncMock(side_effect=stdout_lines)

            mock_subprocess.stderr = AsyncMock()
            mock_subprocess.stderr.readline = AsyncMock(return_value=b"")

            with patch("asyncio.create_subprocess_exec", return_value=mock_subprocess):
                with patch.object(process.logger, "info") as mock_logger:
                    await process.start()

                    # Wait a bit for stdout to be read
                    await asyncio.sleep(0.1)

                    # Verify logger was called with stdout messages
                    calls = [str(call) for call in mock_logger.call_args_list]
                    stdout_calls = [c for c in calls if "ENGINE-STDOUT" in c]
                    self.assertGreater(len(stdout_calls), 0)

            # Clean up tasks
            if process._stdout_task:
                process._stdout_task.cancel()
            if process._stderr_task:
                process._stderr_task.cancel()

        asyncio.run(run_test())

    def test_stderr_capture(self):
        """Test stderr capture and logging."""

        async def run_test():
            process = EngineProcess(self.binary_path, self.args)

            # Create mock subprocess with stderr lines
            mock_subprocess = AsyncMock()
            mock_subprocess.pid = 12345
            mock_subprocess.returncode = None

            mock_subprocess.stdout = AsyncMock()
            mock_subprocess.stdout.readline = AsyncMock(return_value=b"")

            stderr_lines = [
                b"Warning: Low memory\n",
                b"Error loading model\n",
                b"",  # EOF
            ]

            mock_subprocess.stderr = AsyncMock()
            mock_subprocess.stderr.readline = AsyncMock(side_effect=stderr_lines)

            with patch("asyncio.create_subprocess_exec", return_value=mock_subprocess):
                with patch.object(process.logger, "warning") as mock_logger:
                    await process.start()

                    # Wait a bit for stderr to be read
                    await asyncio.sleep(0.1)

                    # Verify logger was called with stderr messages
                    calls = [str(call) for call in mock_logger.call_args_list]
                    stderr_calls = [c for c in calls if "ENGINE-STDERR" in c]
                    self.assertGreater(len(stderr_calls), 0)

            # Clean up tasks
            if process._stdout_task:
                process._stdout_task.cancel()
            if process._stderr_task:
                process._stderr_task.cancel()

        asyncio.run(run_test())

    def test_cleanup_cancels_tasks(self):
        """Test that cleanup cancels log reader tasks."""

        async def run_test():
            process = EngineProcess(self.binary_path, self.args)

            mock_subprocess = AsyncMock()
            mock_subprocess.pid = 12345
            mock_subprocess.returncode = None
            mock_subprocess.stdout = AsyncMock()
            mock_subprocess.stdout.readline = AsyncMock(return_value=b"")
            mock_subprocess.stderr = AsyncMock()
            mock_subprocess.stderr.readline = AsyncMock(return_value=b"")
            mock_subprocess.terminate = Mock()
            mock_subprocess.wait = AsyncMock()

            with patch("asyncio.create_subprocess_exec", return_value=mock_subprocess):
                await process.start()

                stdout_task = process._stdout_task
                stderr_task = process._stderr_task

                self.assertIsNotNone(stdout_task)
                self.assertIsNotNone(stderr_task)

                await process.stop()

                # Tasks should be cancelled
                self.assertIsNone(process._stdout_task)
                self.assertIsNone(process._stderr_task)

        asyncio.run(run_test())

    def test_process_crash_detection(self):
        """Test detection of crashed processes."""

        async def run_test():
            process = EngineProcess(self.binary_path, self.args)

            mock_subprocess = AsyncMock()
            mock_subprocess.pid = 12345
            mock_subprocess.returncode = None
            mock_subprocess.stdout = AsyncMock()
            mock_subprocess.stdout.readline = AsyncMock(return_value=b"")
            mock_subprocess.stderr = AsyncMock()
            mock_subprocess.stderr.readline = AsyncMock(return_value=b"")

            with patch("asyncio.create_subprocess_exec", return_value=mock_subprocess):
                await process.start()

                # Simulate process crash (non-zero exit code)
                mock_subprocess.returncode = 1

                # Check status - should be detected as crashed
                status = process.status
                self.assertEqual(status, "crashed")
                self.assertFalse(process.is_running)

            # Clean up tasks
            if process._stdout_task:
                process._stdout_task.cancel()
            if process._stderr_task:
                process._stderr_task.cancel()

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
