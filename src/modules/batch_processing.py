import numpy as np
import os
import sys

import subprocess
import re
import preprocessing
import segmentation

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.modules.logger import logger

# TODO - Design

if __name__ == "__main__":
    logger.info("Testing batchpipeline")