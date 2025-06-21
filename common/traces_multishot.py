from os import PathLike
from pathlib import Path

from analysislib.common.traces_single_shot import TraceSingleshotAnalysis
from analysislib.common.plot_config import PlotConfig
from .image import Image

from typing import Union
import numpy as np
import matplotlib.pyplot as plt



class TraceMultishotAnalysis():
    """
    Class for analyzing the entire folder
    """

    def __init__(self, folder_path: Union[str, PathLike]):
        self.traces_name, self.traces_time, self.traces_valu_lst = self.analyze_the_folder(folder_path)


    @classmethod
    def analyze_the_folder(cls, h5_path: Union[str, PathLike]):
        sequence_dir = Path(h5_path)
        shots_h5s = sequence_dir.glob('20*.h5')

        print('Loading traces...')
        traces_value_lst = []
        for shot in shots_h5s:
            print(shot)
            trace_single = TraceSingleshotAnalysis(load_type='h5', h5_path=shot)
            traces_name = trace_single.traces_name
            traces_time = trace_single.traces_time
            traces_value = trace_single.traces_value

            traces_value_lst.append(traces_value)

        traces_value_lst = np.array(traces_value_lst)

        return traces_name, traces_time, traces_value_lst



    def plot_traces(self, ax = None):
        '''
        Returns the average background for the entire folder
        The average is calculated by averaging the background (last shot) of each image
        '''
        if ax is None:
            ax = plt.subplots(nrows=1, ncols=1, layout='constrained')
        else:
            ax = ax


        for i in np.arange(len(self.traces_name)):
            traces_value = self.traces_value_lst[:,i,:]
            traces_mean = np.mean(traces_value, axis=0)
            traces_std = np.std(traces_value, axis=0)

            ax.plot(self.traces_time, traces_mean, label = f'{self.traces_name[i]}')
            ax.fill_between(self.traces_time, traces_mean - traces_std, traces_mean + traces_std, alpha=0.2)

        ax.set_title('Traces')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage(V)')
        ax.legend()

        return ax

