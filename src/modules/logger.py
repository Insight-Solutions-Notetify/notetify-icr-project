import logging
import os
import time
from functools import wraps
from typing import Any, Callable

import colorlog

from src.config.logger_config import logger_config


def setup_logger(config=logger_config) -> logging.Logger:
    """
    Setup logger"
    """
    logger = logging.getLogger(config.LOGGER_NAME)
    if logger_config.DEBUG_MODE:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers = []

    # Console Handler with colors
    console_handler = colorlog.StreamHandler()
    color_formatter = colorlog.ColoredFormatter(
        config.COLOR_LOG_FORMAT, log_colors=config.COLOR_SCHEME, reset=True, style="%"
    )
    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)

    # File Handler (without colors)
    os.makedirs(config.LOGS_FOLDER, exist_ok=True)
    file_formatter = logging.Formatter(config.LOG_FORMAT)

    file_handler = logging.FileHandler(
        os.path.join(config.LOGS_FOLDER, config.LOG_FILE_NAME)
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Current session log
    last_path = os.path.join(config.LOGS_FOLDER, config.LAST_FILE_NAME)
    if os.path.exists(last_path):
        os.remove(last_path)

    session_handler = logging.FileHandler(last_path)
    session_handler.setFormatter(file_formatter)
    logger.addHandler(session_handler)

    return logger

def log_execution_time(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    A decorator that logs the execution time of a function.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result

    return wrapper

logger = setup_logger()