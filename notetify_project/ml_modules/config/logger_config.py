import logging
import colorlog

class LoggerConfig:
    LOGGER_NAME = "notetify_icr_cnn"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    COLOR_LOG_FORMAT = (
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s"
    )
    LOGS_FOLDER = "logs"
    LAST_FILE_NAME = "last.log"
    LOG_FILE_NAME = "app.log"

    # Color scheme for different log levels
    COLOR_SCHEME = {
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    }
    
    DEBUG_MODE = True


logger_config = LoggerConfig()