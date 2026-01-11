"""
Environment variable handling for KiCad MCP Server.
"""
import os
import logging
from typing import Dict, Optional

def load_dotenv(env_file: str = ".env") -> Dict[str, str]:
    """Load environment variables from .env file.

    Args:
        env_file: Path to the .env file (can be absolute or relative)

    Returns:
        Dictionary of loaded environment variables
    """
    env_vars = {}
    logging.info(f"load_dotenv called for file: {env_file}")

    # Check if env_file is an absolute path
    if os.path.isabs(env_file):
        env_path = env_file if os.path.exists(env_file) else None
        if env_path:
            logging.info(f"Using absolute path: {env_path}")
    else:
        # Try to find .env file in the current directory or parent directories
        env_path = find_env_file(env_file)
    
    if not env_path:
        logging.warning(f"No .env file found matching: {env_file}")
        return env_vars
    
    logging.info(f"Found .env file at: {env_path}")
    
    try:
        with open(env_path, 'r') as f:
            logging.info(f"Successfully opened {env_path} for reading.")
            line_num = 0
            for line in f:
                line_num += 1
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    logging.debug(f"Skipping line {line_num} (comment/empty): {line}")
                    continue
                
                # Parse key-value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    logging.debug(f"Parsed line {line_num}: Key='{key}', RawValue='{value}'")
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    # Expand ~ to user's home directory
                    original_value = value
                    if '~' in value:
                        value = os.path.expanduser(value)
                        if value != original_value:
                            logging.debug(f"Expanded ~ in value for key '{key}': '{original_value}' -> '{value}'")
                    
                    # Set environment variable
                    logging.info(f"Setting os.environ['{key}'] = '{value}'")
                    os.environ[key] = value
                    env_vars[key] = value
                else:
                    logging.warning(f"Skipping line {line_num} (no '=' found): {line}")
            logging.info(f"Finished processing {env_path}")
            
    except Exception as e:
        # Use logging.exception to include traceback
        logging.exception(f"Error loading .env file '{env_path}'") 
    
    logging.info(f"load_dotenv returning: {env_vars}")
    return env_vars

def find_env_file(filename: str = ".env") -> Optional[str]:
    """Find a .env file in the current directory or parent directories.
    
    Args:
        filename: Name of the env file to find
        
    Returns:
        Path to the env file if found, None otherwise
    """
    current_dir = os.getcwd()
    logging.info(f"find_env_file starting search from: {current_dir}")
    max_levels = 3  # Limit how far up to search
    
    for _ in range(max_levels):
        env_path = os.path.join(current_dir, filename)
        if os.path.exists(env_path):
            return env_path
        
        # Move up one directory
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # We've reached the root
            break
        current_dir = parent_dir
    
    return None

def get_env_list(env_var: str, default: str = "") -> list:
    """Get a list from a comma-separated environment variable.
    
    Args:
        env_var: Name of the environment variable
        default: Default value if environment variable is not set
        
    Returns:
        List of values
    """
    value = os.environ.get(env_var, default)
    if not value:
        return []
    
    # Split by comma and strip whitespace
    items = [item.strip() for item in value.split(",")]
    
    # Filter out empty items
    return [item for item in items if item]
