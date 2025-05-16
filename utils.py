import logging

# Configure basic logging for the application
# This will apply to all loggers unless they have specific handlers/config.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# logger = logging.getLogger(__name__) # Removed as utils.py itself doesn't log

def get_logger(name: str) -> logging.Logger:
    """Retrieve a logger instance with the given name."""
    return logging.getLogger(name) 