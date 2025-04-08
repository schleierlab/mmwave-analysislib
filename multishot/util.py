from pathlib import Path
from tkinter.filedialog import askdirectory


# show dialog box and return the path
def select_data_directory() -> Path:
    '''
    Returns
    -------
    Path
        Path to the selected directory
    '''
    while True:
        try:
            return Path(askdirectory(title='Select data directory for tweezer site detection'))
        except Exception as e:
            raise e
