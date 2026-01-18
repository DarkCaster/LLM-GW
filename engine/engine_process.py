# engine/engine_process.py

import asyncio
import asyncio.subprocess
from typing import Optional, List
from utils.logger import get_logger


class EngineProcess:
    """
    Wrapper for managing a single LLM engine subprocess.
    """

    def __init__(
        self, binary_path: str, args: List[str], work_dir: Optional[str] = None
    ):
        """
        Initialize the engine process wrapper.

        Args:
            binary_path: Path to the engine binary executable
            args: List of command-line arguments to pass to the engine
            work_dir: Working directory for the process (optional)
        """
        self.binary_path = binary_path
        self.args = args
        self.work_dir = work_dir
        self.logger = get_logger(self.__class__.__name__)

        self._process: Optional[asyncio.subprocess.Process] = None
        self._status = "stopped"
        self._stdout_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._start_time: Optional[float] = None

    async def start(self) -> None:
        """
        Start the engine subprocess.

        Raises:
            Exception: If process fails to start
        """
        if self._process is not None and self.is_running():
            self.logger.warning("Process is already running")
            return

        try:
            self.logger.info(
                f"Starting engine process: {self.binary_path} {' '.join(self.args)}"
            )

            # Create subprocess
            self._process = await asyncio.create_subprocess_exec(
                self.binary_path,
                *self.args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.work_dir,
            )

            # Record start time
            self._start_time = asyncio.get_event_loop().time()

            # Set status
            self._status = "running"

            # Start log reader tasks
            self._stdout_task = asyncio.create_task(self._read_stdout())
            self._stderr_task = asyncio.create_task(self._read_stderr())

            self.logger.info(f"Engine process started with PID: {self._process.pid}")

        except Exception as e:
            self.logger.error(f"Failed to start engine process: {e}")
            self._status = "stopped"
            raise

    async def stop(self, timeout: float = 10.0) -> None:
        """
        Stop the engine subprocess gracefully, with forceful termination if needed.

        Args:
            timeout: Maximum time to wait for graceful shutdown before forcing
        """
        if self._process is None:
            self.logger.debug("No process to stop")
            return

        if not self.is_running():
            self.logger.debug("Process is not running")
            self._cleanup()
            return

        try:
            self.logger.info(f"Stopping engine process (PID: {self._process.pid})")

            # Send SIGTERM for graceful shutdown
            try:
                self._process.terminate()
            except ProcessLookupError:
                self.logger.debug("Process already terminated")
                self._cleanup()
                return

            # Wait for process to exit with timeout
            try:
                await asyncio.wait_for(self._process.wait(), timeout=timeout)
                self.logger.info("Engine process terminated gracefully")
            except asyncio.TimeoutError:
                # Timeout expired, force kill
                self.logger.warning(
                    f"Engine process did not terminate within {timeout}s, forcing kill"
                )
                try:
                    self._process.kill()
                    await self._process.wait()
                    self.logger.info("Engine process killed forcefully")
                except ProcessLookupError:
                    self.logger.debug("Process already terminated during kill")

        except Exception as e:
            self.logger.error(f"Error stopping engine process: {e}")
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """
        Clean up process resources and tasks.
        """
        # Cancel log reader tasks
        if self._stdout_task and not self._stdout_task.done():
            self._stdout_task.cancel()
        if self._stderr_task and not self._stderr_task.done():
            self._stderr_task.cancel()

        self._status = "stopped"
        self._stdout_task = None
        self._stderr_task = None

    @property
    def is_running(self) -> bool:
        """
        Check if subprocess is still alive.

        Returns:
            True if process is running, False otherwise
        """
        if self._process is None:
            return False

        # Check if process has terminated
        if self._process.returncode is not None:
            # Process has exited
            if self._status == "running":
                self._status = "crashed" if self._process.returncode != 0 else "stopped"
            return False

        return True

    @property
    def pid(self) -> Optional[int]:
        """
        Get process PID.

        Returns:
            Process PID if running, None otherwise
        """
        if self._process is not None and self.is_running:
            return self._process.pid
        return None

    @property
    def status(self) -> str:
        """
        Get process status.

        Returns:
            Status string: "running", "stopped", or "crashed"
        """
        # Update status based on current process state
        if self._process is not None and self._process.returncode is not None:
            if self._status == "running":
                self._status = "crashed" if self._process.returncode != 0 else "stopped"

        return self._status

    async def _read_stdout(self) -> None:
        """
        Continuously read and log stdout from the subprocess.
        """
        if self._process is None or self._process.stdout is None:
            return

        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break

                line_str = line.decode("utf-8", errors="replace").rstrip()
                if line_str:
                    self.logger.info(f"[ENGINE-STDOUT] {line_str}")
        except asyncio.CancelledError:
            self.logger.debug("stdout reader task cancelled")
        except Exception as e:
            self.logger.error(f"Error reading stdout: {e}")

    async def _read_stderr(self) -> None:
        """
        Continuously read and log stderr from the subprocess.
        """
        if self._process is None or self._process.stderr is None:
            return

        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break

                line_str = line.decode("utf-8", errors="replace").rstrip()
                if line_str:
                    self.logger.warning(f"[ENGINE-STDERR] {line_str}")
        except asyncio.CancelledError:
            self.logger.debug("stderr reader task cancelled")
        except Exception as e:
            self.logger.error(f"Error reading stderr: {e}")
