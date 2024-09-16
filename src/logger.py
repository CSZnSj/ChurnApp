# logger.py
import logging
import sys

def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """Function to set up a logger with the specified name and log file.
    
    Args:
        name (str): The name of the logger (usually __name__ of the module).
        log_file (str, optional): The file to log to. Defaults to None (no file).
        level (logging level, optional): The logging level (INFO, DEBUG, etc.). Defaults to INFO.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    
    # Create a custom logger
    logger = logging.getLogger(name)
    
    # If the logger has already been configured, return it
    if logger.hasHandlers():
        return logger
    
    # Set the minimum log level
    logger.setLevel(level)
    
    # Define log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Add the console handler to the logger
    logger.addHandler(console_handler)
    
    # Optionally add a file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger