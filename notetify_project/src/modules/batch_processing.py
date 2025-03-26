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

# TODO - Given a set of images to run the recognition module, load the images and execute the process for each image one at a time.
# Work orders will be sent as a request to work on "named files" when the request is given (parallel to other process)


if __name__ == "__main__":
    logger.info("Testing batchpipeline")
    # MOCK FUNCTIOn OF LOADING IMG (RANDOM IMG), then viewing image (RANDOM READY IMAGE)

