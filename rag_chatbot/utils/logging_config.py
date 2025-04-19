import logging
import os

def get_logger(name):
    """Get a configured logger for the given name."""
    # Configure root logger if not already configured
    if not logging.getLogger().handlers:
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler('logs/app.log')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    # Return logger for the specific module
    return logging.getLogger(name) 