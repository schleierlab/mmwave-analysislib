import importlib.resources
from pathlib import Path

from analysislib import multishot


USERLIB_PATH = Path('C:/Users/sslab/labscript-suite/userlib')
ROI_CONFIG_PATH = importlib.resources.files(multishot) / 'roi_config.yml'
