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

## MAIN CONTROLLER TO RUN the BATCH PIPELINE
## ACTUAL IMPLEMENTATION OF loading, viewing, closing

## PROVIDE COMMANDS to load image to be converted, view images done processing and quit the program


def loadImage():
    pass

def viewImage():
    pass

def close():
    pass

def controller():
    pass

if __name__ == "__main__":
    logger.debug("Running controlller")