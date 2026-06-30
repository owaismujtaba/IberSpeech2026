"""
Central logger for the UGR-MINDVOICE pipeline.
Call get_logger() anywhere to get the same logger instance.
Writes to both stdout and logs/pipeline.log.
"""
import logging
import os
import sys
from datetime import datetime


def get_logger(name: str = "mindvoice") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:          # already configured — return as-is
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", "pipeline.log")
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("=" * 70)
    logger.info(f"Pipeline logger initialised  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {os.path.abspath(log_file)}")
    logger.info("=" * 70)

    return logger
