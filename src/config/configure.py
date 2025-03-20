import sys
import os
import logging

logger = logging.getLogger(__name__)

# Get the user's home directory dynamically
home_dir = os.path.expanduser("~")

if sys.platform.startswith("win"):  # Covers Windows
    logger.info("Running on Windows")
    base_dir = os.path.join(home_dir, "OneDrive", "Documents", "GitHub", "notetify-icr-project")

elif sys.platform.startswith("linux") or sys.platform.startswith("darwin"):  # Covers Linux & Mac
    logger.info("Running on Linux/Mac")
    base_dir = os.path.join(home_dir, "notetify-icr-project")  # Adjust if needed

else:
    logger.warning("Unknown platform detected: %s", sys.platform)
    base_dir = None

# Debugging output
logger.info(f"Base directory set to: {base_dir}")
print(f"Base directory: {base_dir}")  # Print for debugging