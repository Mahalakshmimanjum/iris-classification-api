import logging
from logging.handlers import RotatingFileHandler

from app.config import settings


def setup_logging():
    logger = logging.getLogger("ml_api")

    log_level = getattr(
        logging,
        settings.LOG_LEVEL.upper(),
        logging.INFO
    )

    logger.setLevel(log_level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        "app.log",
        maxBytes=1_000_000,
        backupCount=3
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger