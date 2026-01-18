"""
EngineProcess - Wrapper for managing a single LLM engine subprocess.

Provides process lifecycle management, status monitoring, and log capture.
"""

import asyncio
import logging
import time
from typing import List, Optional

from utils.logger import get_logger


class EngineProcess:
    """Wrapper for managing a single LLM engine subprocess."""

    def __init__(
        self,
        binary_path: str,
        args: List[str],
        work_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize EngineProcess with binary and arguments.

        Args:
            binary_path: Path to the engine binary executable
            args: List of command-line arguments for the binary
            work_dir: Working directory for the subprocess (optional)
            logger: Logger instance (uses class name if not provided)
        """
        self.binary_path = binary_path
        self.args = args
        self.work_dir = work_dir
        self._logger = logger or get_logger(self.__class__.__name__)

        self._process: Optional[asyncio.subprocess.Process] = None
        self._stdout_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._status: str = "initialized"
        self._start_time: Optional[float] = None
        self._stopped_by_us: bool = False
        self._exit_code: Optional[int] = None

    async def start(self) -> None:
        """
        Start the engine subprocess.

        Raises:
            RuntimeError: If process is already running or fails to start
            FileNotFoundError: If binary path does not exist
        """
        if self.is_running():
            raise RuntimeError("Process is already running")

        try:
            # Create subprocess
            self._process = await asyncio.create_subprocess_exec(
                self.binary_path,
                *self.args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.work_dir,
            )

            self._start_time = time.time()
            self._status = "running"
            self._stopped_by_us = False
            self._exit_code = None

            # Start log reading tasks
            self._stdout_task = asyncio.create_task(self._read_stdout())
            self._stderr_task = asyncio.create_task(self._read_stderr())

            self._logger.info(
                f"Started engine process (PID: {self.get_pid()}) "
                f"with command: {self.binary_path} {' '.join(self.args)}"
            )

        except FileNotFoundError:
            self._logger.error(f"Binary not found: {self.binary_path}")
            raise
        except Exception as e:
            self._logger.error(f"Failed to start engine process: {e}")
            self._status = "failed"
            raise RuntimeError(f"Failed to start engine process: {e}")

    async def stop(self, timeout: float = 10.0) -> None:
        """
        Stop the engine process gracefully, with forceful fallback.

        Args:
            timeout: Time to wait for graceful shutdown before forcing (seconds)
        """
        if not self.is_running():
            self._logger.debug("Process is not running, nothing to stop")
            return

        self._logger.info(f"Stopping engine process (PID: {self.get_pid()})...")
        self._stopped_by_us = True

        try:
            # Send SIGTERM for graceful shutdown
            if self._process and self._process.returncode is None:
                self._process.terminate()

            # Wait for process to exit
            try:
                await asyncio.wait_for(self._process.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                self._logger.warning(
                    f"Process did not terminate gracefully within {timeout}s, forcing..."
                )
                if self._process and self._process.returncode is None:
                    self._process.kill()
                    await self._process.wait()

            # Cancel and wait for reader tasks
            if self._stdout_task and not self._stdout_task.done():
                self._stdout_task.cancel()
            if self._stderr_task and not self._stderr_task.done():
                self._stderr_task.cancel()

            # Wait for tasks to complete
            if self._stdout_task:
                try:
                    await self._stdout_task
                except asyncio.CancelledError:
                    pass
            if self._stderr_task:
                try:
                    await self._stderr_task
                except asyncio.CancelledError:
                    pass

            if self._process:
                self._exit_code = self._process.returncode

            self._status = "stopped"
            self._logger.info(f"Engine process stopped (exit code: {self._exit_code})")

        except Exception as e:
            self._logger.error(f"Error during process stop: {e}")
            self._status = "failed"
            raise

    def is_running(self) -> bool:
        """
        Check if the process is currently running.

        Returns:
            True if process is running, False otherwise
        """
        if not self._process:
            return False
        return self._process.returncode is None

    def get_pid(self) -> Optional[int]:
        """
        Get the process ID.

        Returns:
            Process PID if running, None otherwise
        """
        if not self._process:
            return None
        return self._process.pid

    def get_status(self) -> str:
        """
        Get the current process status.

        Returns:
            Status string: "initialized", "running", "stopped", "crashed", "failed"
        """
        if self._status == "running" and not self.is_running():
            # Check if process crashed
            if self._process and self._process.returncode is not None:
                if self._stopped_by_us:
                    self._status = "stopped"
                else:
                    self._status = "crashed"
        return self._status

    def get_uptime(self) -> Optional[float]:
        """
        Get process uptime in seconds.

        Returns:
            Uptime in seconds if running, None otherwise
        """
        if self._start_time and self.is_running():
            return time.time() - self._start_time
        return None

    async def _read_stdout(self) -> None:
        """Continuously read and log stdout from the process."""
        if not self._process or not self._process.stdout:
            return

        try:
            while self.is_running():
                line = await self._process.stdout.readline()
                if not line:
                    break

                decoded_line = line.decode("utf-8", errors="replace").rstrip()
                if decoded_line:  # Skip empty lines
                    self._logger.info(f"[stdout] {decoded_line}")

        except asyncio.CancelledError:
            self._logger.debug("Stdout reader task cancelled")
            raise
        except Exception as e:
            self._logger.error(f"Error reading stdout: {e}")
        finally:
            self._logger.debug("Stdout reader task finished")

    async def _read_stderr(self) -> None:
        """Continuously read and log stderr from the process."""
        if not self._process or not self._process.stderr:
            return

        try:
            while self.is_running():
                line = await self._process.stderr.readline()
                if not line:
                    break

                decoded_line = line.decode("utf-8", errors="replace").rstrip()
                if decoded_line:  # Skip empty lines
                    self._logger.warning(f"[stderr] {decoded_line}")

        except asyncio.CancelledError:
            self._logger.debug("Stderr reader task cancelled")
            raise
        except Exception as e:
            self._logger.error(f"Error reading stderr: {e}")
        finally:
            self._logger.debug("Stderr reader task finished")

    def __str__(self) -> str:
        """String representation of the engine process."""
        status = self.get_status()
        pid = self.get_pid() or "N/A"
        uptime = self.get_uptime()
        uptime_str = f"{uptime:.1f}s" if uptime else "N/A"

        return (
            f"EngineProcess(pid={pid}, status={status}, "
            f"uptime={uptime_str}, binary={self.binary_path})"
        )
