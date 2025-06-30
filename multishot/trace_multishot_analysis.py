
import matplotlib.pyplot as plt
from analysislib.multishot.util import select_data_directory
from analysislib.common.traces_multishot import TraceMultishotAnalysis
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

folder = select_data_directory()

multishot_analysis = TraceMultishotAnalysis(folder)

fig, ax = plt.subplots(nrows=1, ncols=1, layout='constrained')
multishot_analysis.plot_traces(ax=ax)

