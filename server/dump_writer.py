import os
import datetime
from typing import Optional
from logger import get_logger


class DumpWriter:
    """
    Utility class for writing request/response dumps to files.
    Writes data incrementally to prevent loss on premature termination.
    """

    def __init__(self, dumps_dir: str, model_name: Optional[str] = None):
        """
        Initialize DumpWriter and create dump file.

        Args:
            dumps_dir: Directory where dump files should be written
            model_name: Optional model name to include in filename
        """
        self.logger = get_logger(self.__class__.__name__)
        self._dumps_dir = dumps_dir
        self._model_name = model_name or "unknown"
        self._file = None
        self._filepath = None
        self._is_closed = False

        # Generate filename with timestamp
        self._filepath = self._generate_filename()

        # Open file for writing
        try:
            self._file = open(self._filepath, "w", encoding="utf-8")
            self.logger.debug(f"Created dump file: {self._filepath}")
        except Exception as e:
            self.logger.error(f"Failed to create dump file {self._filepath}: {e}")
            self._file = None

    def _generate_filename(self) -> str:
        """
        Generate filename with timestamp and model name.

        Returns:
            Full path to the dump file
        """
        now = datetime.datetime.now()
        # Format: YYYY-MM-DD_HH-mm-ss-msec_[modelname].dump.txt
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        milliseconds = now.microsecond // 1000
        filename = f"{timestamp}-{milliseconds:03d}_{self._model_name}.dump.txt"
        return os.path.join(self._dumps_dir, filename)

    def write_request(self, request_text: str) -> None:
        """
        Write request section to dump file.

        Args:
            request_text: Raw request text to write
        """
        if self._file is None or self._is_closed:
            return

        try:
            separator = "=" * 80
            self._file.write(f"{separator}\n")
            self._file.write("REQUEST\n")
            self._file.write(f"{separator}\n")
            self._file.write(request_text)
            if not request_text.endswith("\n"):
                self._file.write("\n")
            self._file.write(f"{separator}\n\n")
            self._file.flush()
            self.logger.debug(f"Wrote request to dump file ({len(request_text)} chars)")
        except Exception as e:
            self.logger.error(f"Failed to write request to dump file: {e}")

    def write_response(self, response_text: str) -> None:
        """
        Write response section to dump file.

        Args:
            response_text: Response text to write
        """
        if self._file is None or self._is_closed:
            return

        try:
            separator = "=" * 80
            self._file.write(f"{separator}\n")
            self._file.write("RESPONSE\n")
            self._file.write(f"{separator}\n")
            self._file.write(response_text)
            if not response_text.endswith("\n"):
                self._file.write("\n")
            self._file.write(f"{separator}\n\n")
            self._file.flush()
            self.logger.debug(
                f"Wrote response to dump file ({len(response_text)} chars)"
            )
        except Exception as e:
            self.logger.error(f"Failed to write response to dump file: {e}")

    def write_response_chunk(self, chunk: bytes) -> None:
        """
        Write a chunk of streaming response to dump file.

        Args:
            chunk: Response chunk bytes to write
        """
        if self._file is None or self._is_closed:
            return

        try:
            # Decode chunk and write it
            chunk_text = chunk.decode("utf-8", errors="replace")
            self._file.write(chunk_text)
            self._file.flush()
        except Exception as e:
            self.logger.error(f"Failed to write response chunk to dump file: {e}")

    def write_response_start(self) -> None:
        """
        Write response section header for streaming responses.
        """
        if self._file is None or self._is_closed:
            return

        try:
            separator = "=" * 80
            self._file.write(f"{separator}\n")
            self._file.write("RESPONSE (STREAMING)\n")
            self._file.write(f"{separator}\n")
            self._file.flush()
            self.logger.debug("Wrote response header to dump file")
        except Exception as e:
            self.logger.error(f"Failed to write response header to dump file: {e}")

    def write_response_end(self) -> None:
        """
        Write response section footer for streaming responses.
        """
        if self._file is None or self._is_closed:
            return

        try:
            separator = "=" * 80
            self._file.write(f"\n{separator}\n\n")
            self._file.flush()
            self.logger.debug("Wrote response footer to dump file")
        except Exception as e:
            self.logger.error(f"Failed to write response footer to dump file: {e}")

    def write_error(self, error: Exception) -> None:
        """
        Write error/exception section to dump file.

        Args:
            error: Exception object to write
        """
        if self._file is None or self._is_closed:
            return

        try:
            import traceback

            separator = "=" * 80
            self._file.write(f"{separator}\n")
            self._file.write("ERROR\n")
            self._file.write(f"{separator}\n")
            self._file.write(f"Exception Type: {type(error).__name__}\n")
            self._file.write(f"Exception Message: {str(error)}\n")
            self._file.write("\nTraceback:\n")
            self._file.write(traceback.format_exc())
            self._file.write(f"{separator}\n\n")
            self._file.flush()
            self.logger.debug("Wrote error to dump file")
        except Exception as e:
            self.logger.error(f"Failed to write error to dump file: {e}")

    def close(self) -> None:
        """
        Close the dump file.
        """
        if self._file is not None and not self._is_closed:
            try:
                self._file.close()
                self._is_closed = True
                self.logger.debug(f"Closed dump file: {self._filepath}")
            except Exception as e:
                self.logger.error(f"Failed to close dump file: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def __del__(self):
        """Destructor to ensure file is closed."""
        if not self._is_closed:
            self.close()
