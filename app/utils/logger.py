import logging
import confuse
from pathlib import Path
from datetime import datetime
from rich.logging import RichHandler


def load_logging_config():
    """Load logging configuration using confuse"""
    config = confuse.Configuration("lora_training")
    config.set_file(Path(__file__).parent.parent / "config" / "logging_config.yaml")
    return config["logging"].get()


def setup_logging():
    """
    Set up logging configuration for the application using settings from config file.
    This should be called once at application startup.
    """
    config = load_logging_config()

    # Create formatters
    # console_formatter = logging.Formatter(config["format"])
    file_formatter = logging.Formatter(config["file_format"])

    # Create handlers
    handlers = []

    # Add console handler if enabled
    if config["handlers"]["console"]:
        console_handler = RichHandler(
            rich_tracebacks=True, markup=True, show_time=True, show_path=True
        )
        handlers.append(console_handler)

    # Add file handler if enabled
    if config["handlers"]["file"]:
        log_dir = Path(config["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)

        # Replace timestamp in filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / config["file_name"].format(timestamp=timestamp)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(level=getattr(logging, config["level"]), handlers=handlers)

    # Set specific log levels for libraries
    for logger_name, level in config["library_log_levels"].items():
        logging.getLogger(logger_name).setLevel(getattr(logging, level))

    # Create logger for this module
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")
    logger.info(
        f"Log file: {log_file if config['handlers']['file'] else 'No file logging'}"
    )

    return logger


def get_logger(name=__name__):
    """
    Get a logger instance with the specified name.

    Args:
        name (str): The name for the logger

    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)


# Initialize logging when this module is imported
setup_logging()
