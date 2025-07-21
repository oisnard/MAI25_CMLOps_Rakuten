import os 
import logging
import src.tools.tools as tools
import shutil


def get_current_type_datastream() -> str:
    """
    Get the current datastream for logging.
    """
    if not os.path.exists(tools.DATA_PROCESSED_STREAM_DIR):
        logging.error(f"{tools.DATA_PROCESSED_STREAM_DIR} folder does not exist.")
        raise FileNotFoundError(f"{tools.DATA_PROCESSED_STREAM_DIR} folder does not exist.")
    X_files = sorted(f for f in os.listdir(tools.DATA_PROCESSED_STREAM_DIR) if f.startswith("X_") and f.endswith(".csv"))
    if not X_files:
        logging.error(f"No X datastream files found in {tools.DATA_PROCESSED_STREAM_DIR}.")
        raise FileNotFoundError(f"No X datastream files found in {tools.DATA_PROCESSED_STREAM_DIR}.")
    if len(X_files) == 1:
        return "baseline"
    else:
        return "stream"
    