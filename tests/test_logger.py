import unittest
import logging
import io
from unittest.mock import patch
from utils.logger import setup_logging, get_logger


class TestLogger(unittest.TestCase):
    def setUp(self):
        """Reset logging configuration before each test."""
        # Remove all handlers from root logger
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.root.setLevel(logging.WARNING)  # Reset to default

    def test_get_logger_returns_correct_logger(self):
        """Test that get_logger returns a logger with the correct name."""
        logger_name = "test.module"
        logger = get_logger(logger_name)

        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, logger_name)

        # Verify it's not the root logger
        self.assertNotEqual(logger.name, "root")

    def test_get_logger_same_name_returns_same_instance(self):
        """Test that calling get_logger with same name returns same instance."""
        logger1 = get_logger("test.module")
        logger2 = get_logger("test.module")

        self.assertIs(logger1, logger2)

    def test_setup_logging_default_configuration(self):
        """Test default logging setup with console handler."""
        # Capture stdout to check log output
        captured_output = io.StringIO()

        with patch("sys.stdout", captured_output):
            setup_logging()
            logger = get_logger("test.logger")

            # Test different log levels
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")

            output = captured_output.getvalue()

            # Debug should not appear (default level is INFO)
            self.assertNotIn("Debug message", output)

            # Info and above should appear
            self.assertIn("Info message", output)
            self.assertIn("Warning message", output)
            self.assertIn("Error message", output)

            # Check format includes expected components
            self.assertIn("test.logger", output)
            self.assertIn("INFO", output)
            self.assertIn("WARNING", output)
            self.assertIn("ERROR", output)

    def test_setup_logging_custom_level(self):
        """Test logging setup with custom log level."""
        captured_output = io.StringIO()

        with patch("sys.stdout", captured_output):
            setup_logging(level=logging.WARNING)
            logger = get_logger("test.logger")

            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")

            output = captured_output.getvalue()

            # Info should not appear (level is WARNING)
            self.assertNotIn("Info message", output)

            # Warning and error should appear
            self.assertIn("Warning message", output)
            self.assertIn("Error message", output)

    def test_setup_logging_custom_format(self):
        """Test logging setup with custom format string."""
        custom_format = "%(levelname)s: %(message)s"
        captured_output = io.StringIO()

        with patch("sys.stdout", captured_output):
            setup_logging(format_string=custom_format)
            logger = get_logger("test.logger")
            logger.info("Test message")

            output = captured_output.getvalue()

            # Check custom format
            self.assertIn("INFO: Test message", output)
            # Default format elements should not be present
            self.assertNotIn("test.logger", output)

    def test_logger_inherits_parent_configuration(self):
        """Test that child loggers inherit configuration from root."""
        setup_logging(level=logging.INFO)

        root_logger = logging.getLogger()
        child_logger = get_logger("parent.child")

        # Both should have same effective level
        self.assertEqual(root_logger.getEffectiveLevel(), logging.INFO)
        self.assertEqual(child_logger.getEffectiveLevel(), logging.INFO)

    def test_multiple_handlers_not_duplicated(self):
        """Test that setup_logging doesn't create duplicate handlers."""
        # First setup
        setup_logging()
        root_logger = logging.getLogger()
        initial_handler_count = len(root_logger.handlers)

        # Second setup should not add duplicate handlers
        setup_logging()
        final_handler_count = len(root_logger.handlers)

        self.assertEqual(initial_handler_count, final_handler_count)

    def test_logger_propagates_to_root(self):
        """Test that loggers propagate messages to root handler."""
        captured_output = io.StringIO()

        with patch("sys.stdout", captured_output):
            setup_logging()
            child_logger = get_logger("parent.child")

            # Log a message using child logger
            child_logger.info("Child logger message")

            output = captured_output.getvalue()

            # Message should appear even though logged by child
            self.assertIn("Child logger message", output)
            # Should show child logger name in output
            self.assertIn("parent.child", output)


if __name__ == "__main__":
    unittest.main()
