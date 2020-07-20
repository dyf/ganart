import glob
import os
import imageio
import numpy as np
import random
import pandas as pd
import re
import tensorflow as tf
import h5py

DEFAULT_H5_PATH = "data/charts/charts.h5"
RESOLUTIONS = {
    512: 'chart_512',
    256: 'chart_256',
    128: 'chart_128',
    64: 'chart_64',
    32: 'chart_32',
    16: 'chart_16',
    8: 'chart_8',
}

def iterdata(resolution, batch_size=10, h5_file=DEFAULT_H5_PATH, shuffle=True, random_seed=None, downsample_factor=None):    

    with h5py.File(DEFAULT_H5_PATH,'r') as hf:
        ds_name = RESOLUTIONS[resolution]
        num_images = hf[ds_name].shape[0]

        num_batches = num_images // batch_size
        inds = list(range(num_images))
        
        for i in range(num_batches):            
            if shuffle:
                rows = sorted(random.sample(inds, k=batch_size))
            else:
                rows = slice(i*batch_size,(i+1)*batch_size)

            yield hf['chart_types'][rows], (hf[ds_name][rows] - 127.5) / 127.5


if __name__ == "__main__":
    for md, images in iterdata(resolution=8):
        print(md.shape, images.shape)
        break