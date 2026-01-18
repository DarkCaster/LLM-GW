# engine/engine_process.py

import asyncio
import asyncio.subprocess
import logging
from typing import List, Optional
import time


class EngineProcess:
    """
    Wrapper for managing a single LLM engine subprocess.
    """

    def __init__(
        self, binary_path: str, args: List[str], work_dir: Optional[str] = None
    ):
        """
        Initialize EngineProcess.

        Args:
            binary_path: Path to the engine binary executable
            args: List of command-line arguments for the binary
            work_dir: Working directory for the process (optional)
        """
        self.binary_path = binary_path
        self.args = args
        self.work_dir = work_dir
        self.logger = logging.getLogger(self.__class__.__name__)

        self._process: Optional[asyncio.subprocess.Process] = None
        self._status = "stopped"
        self._start_time: Optional[float] = None
        self._stdout_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """
        Start the engine subprocess.

        Spawns the engine binary with configured arguments and sets up
        logging of stdout/stderr.
        """
        if self._process is not None and self.is_running():
            self.logger.warning("Process is already running")
            return

        self.logger.info(
            f"Starting engine process: {self.binary_path} with args: {self.args}"
        )

        try:
            # Spawn the subprocess
            self._process = await asyncio.subprocess.create_subprocess_exec(
                self.binary_path,
                *self.args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.work_dir,
            )

            self._start_time = time.time()
            self._status = "running"

            # Create background tasks to read and log output
            self._stdout_task = asyncio.create_task(self._read_stdout())
            self._stderr_task = asyncio.create_task(self._read_stderr())

            self.logger.info(f"Engine process started with PID: {self._process.pid}")

        except Exception as e:
            self.logger.error(f"Failed to start engine process: {e}")
            self._status = "crashed"
            raise

    async def stop(self, timeout: float = 10.0) -> None:
        """
        Stop the engine subprocess.

        Attempts graceful shutdown with SIGTERM, then forceful with SIGKILL if needed.

        Args:
            timeout: Maximum time to wait for graceful shutdown before using SIGKILL
        """
        if self._process is None:
            self.logger.debug("No process to stop")
            return

        if not self.is_running():
            self.logger.debug("Process is not running")
            self._cleanup()
            return

        self.logger.info(f"Stopping engine process (PID: {self._process.pid})")

        try:
            # Send SIGTERM for graceful shutdown
            self._process.terminate()
            self.logger.debug(f"Sent SIGTERM to process {self._process.pid}")

            # Wait for process to exit with timeout
            try:
                await asyncio.wait_for(self._process.wait(), timeout=timeout)
                self.logger.info(f"Process {self._process.pid} terminated gracefully")
            except asyncio.TimeoutError:
                # Timeout expired, send SIGKILL for forceful shutdown
                self.logger.warning(
                    f"Process {self._process.pid} did not terminate within {timeout}s, sending SIGKILL"
                )
                self._process.kill()
                await self._process.wait()
                self.logger.info(f"Process {self._process.pid} killed forcefully")

        except ProcessLookupError:
            # Process already terminated
            self.logger.debug("Process already terminated")
        except Exception as e:
            self.logger.error(f"Error stopping process: {e}")
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """Clean up process resources and background tasks."""
        # Cancel stdout/stderr reader tasks
        if self._stdout_task is not None and not self._stdout_task.done():
            self._stdout_task.cancel()
        if self._stderr_task is not None and not self._stderr_task.done():
            self._stderr_task.cancel()

        self._status = "stopped"
        self._process = None
        self._stdout_task = None
        self._stderr_task = None

    @property
    def is_running(self) -> bool:
        """
        Check if subprocess is still alive.

        Returns:
            True if running, False otherwise
        """
        if self._process is None:
            return False

        # Check if process has exited
        if self._process.returncode is not None:
            # Process has exited
            if self._status == "running":
                # Mark as crashed if it exited unexpectedly
                self._status = "crashed"
            return False

        return True

    @property
    def get_pid(self) -> Optional[int]:
        """
        Get process PID if running.

        Returns:
            Process PID if running, None otherwise
        """
        if self._process is not None and self.is_running:
            return self._process.pid
        return None

    @property
    def get_status(self) -> str:
        """
        Get process status.

        Returns:
            Status string: "running", "stopped", or "crashed"
        """
        # Update status based on current process state
        if self._process is not None and self._process.returncode is not None:
            # Process has exited
            if self._status == "running":
                # Was running but exited unexpectedly
                self._status = "crashed"

        return self._status

    async def _read_stdout(self) -> None:
        """
        Continuously read stdout and log with INFO level.

        Runs as a background task while process is alive.
        """
        if self._process is None or self._process.stdout is None:
            return

        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    # EOF reached
                    break

                # Decode and log the line
                line_str = line.decode("utf-8", errors="replace").rstrip()
                if line_str:  # Only log non-empty lines
                    self.logger.info(f"[STDOUT] {line_str}")

        except asyncio.CancelledError:
            self.logger.debug("Stdout reader task cancelled")
        except Exception as e:
            self.logger.error(f"Error reading stdout: {e}")

    async def _read_stderr(self) -> None:
        """
        Continuously read stderr and log with WARNING level.

        Runs as a background task while process is alive.
        """
        if self._process is None or self._process.stderr is None:
            return

        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    # EOF reached
                    break

                # Decode and log the line
                line_str = line.decode("utf-8", errors="replace").rstrip()
                if line_str:  # Only log non-empty lines
                    self.logger.warning(f"[STDERR] {line_str}")

        except asyncio.CancelledError:
            self.logger.debug("Stderr reader task cancelled")
        except Exception as e:
            self.logger.error(f"Error reading stderr: {e}")
