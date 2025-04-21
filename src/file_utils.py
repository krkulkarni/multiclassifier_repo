# src/file_utils.py
import pickle
from pathlib import Path
from typing import Any, Union
import logging

def pickle_file(data_to_save: Any, file_path: Union[str, Path]):
    """Saves data to a pickle file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    try:
        with open(file_path, "wb") as f:
            pickle.dump(data_to_save, f)
        logging.info(f"Successfully saved data to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save pickle file {file_path}: {e}", exc_info=True)
        raise # Re-raise the exception after logging

def load_pickle(file_path: Union[str, Path]) -> Any:
    """Loads data from a pickle file."""
    file_path = Path(file_path)
    if not file_path.is_file():
        logging.error(f"Pickle file not found: {file_path}")
        raise FileNotFoundError(f"No such file: '{file_path}'")
    try:
        with open(file_path, "rb") as f:
            loaded_data = pickle.load(f)
        logging.info(f"Successfully loaded data from {file_path}")
        return loaded_data
    except Exception as e:
        logging.error(f"Failed to load pickle file {file_path}: {e}", exc_info=True)
        raise # Re-raise the exception after logging