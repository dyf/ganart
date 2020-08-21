import h5py
import random
import numpy as np
import glob
import os

TRAIN_FILE_PATTERN = 'data/charts/bars_train_*.h5'
TEST_FILE_PATTERN = 'data/charts/bars_test_*.h5'


def iterdata(file_pattern=TRAIN_FILE_PATTERN, batch_size=10):
    files = [ os.path.normpath(f) for f in glob.glob(file_pattern) ]
    
    random.shuffle(files)
    
    for file_name in files:
        print(file_name)
        with h5py.File(file_name, 'r') as hf:
            x = hf['x']
            y = hf['y']
            chart = hf['chart']
            ori = hf['ori']
            color = hf['color']

            num_rows = chart.shape[0]    
            num_batches = num_rows // batch_size - 1

            inds = list(range(num_rows))

            for bi in range(num_batches):
                row_inds = sorted(random.sample(inds, k=batch_size))

                xr, yr = x[row_inds], y[row_inds]

               

                #yield x[row_inds], y[row_inds], ((chart[row_inds] - 127.5) / 127.5).astype(np.float32), ori[row_inds][:,np.newaxis], color[row_inds]
                yield xr, yr, (chart[row_inds] / 255.0).astype(np.float32), ori[row_inds][:,np.newaxis], color[row_inds]
    

if __name__ == "__main__":
    x,y,chart = next(iterdata())
    import matplotlib.pyplot as plt
    plt.imshow(chart[0])
    plt.show()
    