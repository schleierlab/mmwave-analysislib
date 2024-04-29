import logging

import h5py
import lyse
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


logging.basicConfig(
    format='%(asctime)s :: %(levelname)s :: %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S %z',
)
logging.captureWarnings(True)

if lyse.spinning_top:
    h5_path = lyse.path
else:
    df = lyse.data()
    h5_path = df.filepath.iloc[-1]

# wait_time = 1.5
# print(f'Waiting for {wait_time} seconds')
# time.sleep(wait_time)

images = np.load(r'C:\labscript-suite\tmp\kinetix.npy')

with h5py.File(h5_path, mode='r+') as f:
    exist_kinetix_exposures = ('EXPOSURES' in f['devices/kinetix'])
    if exist_kinetix_exposures:
        camera_group = f['devices']['kinetix']
        image_group = f.require_group('/images/kinetix')

        for i, image in enumerate(images):
            image_group.create_dataset(str(i), data=image, dtype='int16')
            logging.info(f'Image {i} saved')

if exist_kinetix_exposures:
    logging.info('Plotting Kinetix images')

    fig, ax = plt.subplots(figsize=(15,15))

    im = ax.imshow(image, cmap='viridis')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax)
