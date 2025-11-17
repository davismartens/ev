# app/utils/logger.py
import logging
import sys

def get_logger(name: str = "app", level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger that can be safely imported anywhere.
    Avoids duplicate handlers when imported multiple times.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# default shared logger instance
logger = get_logger("blog_service")
