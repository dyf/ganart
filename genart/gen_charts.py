import os
import matplotlib.pyplot as plt
import numpy as np
import random
from random_words import RandomWords
import sklearn.datasets
import enum
import pandas as pd
import h5py 
from PIL import Image

@enum.unique
class Chart(enum.Enum):
    BAR = 0
    LINE = 1
    SCATTER = 2


CMAPS = [ 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 
          'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
          'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
          'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
          'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper',
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
            'twilight', 'twilight_shifted', 'hsv',
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c',
                        'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']

MARKERS = [ ".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_",0,1,2,3,4,5,6,7,8,9,10,11]

LEGEND_LOCS = ['upper left', 'upper right', 'lower left', 'lower right' ]

FIGSIZE = (5.12,5.12)

def gen_bar(max_bars=50, data_scale=100, figsize=FIGSIZE):
    data_scale = np.random.random()*data_scale
    n_rows = random.randint(1, max_bars)
    data = np.random.random(n_rows) * data_scale
    
    horizontal = np.random.random() < 0.5
    categorical = np.random.random() < 0.5
    encode_color = np.random.random() < 0.5
    
    barf = plt.barh if horizontal else plt.bar
    tickf = plt.yticks if horizontal else plt.xticks

    ticks = None
    if categorical:
        rw = RandomWords()
        ticks = rw.random_words(count=n_rows)

    if encode_color:
        color = np.random.random((n_rows,3))
    else:
        color = np.random.random((1,3))        

    fig, ax = plt.subplots(figsize=figsize)

    coords = range(data.shape[0]) 
    barf(coords, data, tick_label=ticks, color=color)
    if not horizontal and categorical:
        plt.xticks(rotation=90)
    plt.tight_layout()
    
def gen_scatter(max_rows_per_var=400, data_scale=100, max_vars=5, enc_size_p=0.5, enc_color_p=0.5, uniform_dist_p=0.0, marker_max_size=200, figsize=FIGSIZE):
    data_scale = np.random.random()*data_scale
    num_vars = random.randint(1, max_vars)    
    labels = RandomWords().random_words(count=num_vars)

    fig, ax = plt.subplots(figsize=figsize)

    for vi in range(num_vars):
        num_rows = random.randint(1, max_rows_per_var)
        
        if np.random.random() < uniform_dist_p: # uniform random
            data = (np.random.random((num_rows,2)) * 2 - 1) * data_scale
        else: 
            # gaussian
            cov = sklearn.datasets.make_spd_matrix(2)            
            mean = np.random.random(2)
            data = (np.random.multivariate_normal(mean, cov, size=num_rows) * 2 - 1) * data_scale

        marker = random.choice(MARKERS)
        cmap = None

        if np.random.random() < enc_size_p:
            size = np.random.random(num_rows) * marker_max_size            
        else: 
            size = np.random.random() * marker_max_size
        
        if np.random.random() < enc_color_p:
            color = np.random.random(num_rows)
            cmap = random.choice(CMAPS)
        else:
            color = np.random.random((1,3))

        ax.scatter(data[:,0], data[:,1], marker=marker, s=size, c=color, cmap=cmap, label=labels[vi])    
    plt.legend()
    plt.tight_layout()

def gen_line(max_rows=200, max_vars=5, data_scale=100, figsize=FIGSIZE):

    data_scale = np.random.random()*data_scale
    num_vars = random.randint(1, max_vars)    
    labels = RandomWords().random_words(count=num_vars)
    num_rows = random.randint(1, max_rows)
    
    cov = sklearn.datasets.make_spd_matrix(num_vars)
    mean = np.random.random(num_vars)
        
    data = (np.random.multivariate_normal(mean, cov, size=num_rows) * 2 - 1 ) * data_scale

    tr = np.random.random(2)
    tr = tr if tr[0] < tr[1] else [ tr[1], tr[0] ]
    t = np.linspace(tr[0], tr[1], num_rows)

    fig, ax = plt.subplots(figsize=figsize)

    for vi in range(num_vars):
        color = np.random.random(3)
        ax.plot(t, data[:,vi], c=color, label=labels[vi])    

    plt.legend()
    plt.tight_layout()
    
def gen_images(N, out_dir, out_h5):
    
    chart_fs = {
        Chart.BAR: gen_bar,
        Chart.LINE: gen_line,
        Chart.SCATTER: gen_scatter
    }

    out = []

    charts = random.choices(list(chart_fs.keys()), k=N)
    chart_types = [ c.name for c in charts ]
    chart_idxs = [ c.value for c in charts ]
    
    sizes = [ 512, 256, 128, 64, 32, 16, 8 ]

    with h5py.File(out_h5, 'w') as f:
        #f.create_dataset('chart_types', data=chart_types)
        f.create_dataset('chart_idxs', data=chart_idxs)

        img_hs = [ f.create_dataset(f'chart_{s}', shape=(N,s,s,3), dtype='uint8') for s in sizes ]


        for i in range(N):
            if i % 10 == 0:
                print(i)

            chart = charts[i]
            chart_fs[chart]()

            fig = plt.gcf()
            fig.canvas.draw()    
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))                    
            
            for si, size in enumerate(sizes):
                size_data = data
                if si > 0:
                    size_data = np.array(Image.fromarray(data).resize((size,size)))
                
                img_hs[si][i,:,:,:] = size_data

            plt.close()

        


if __name__ == "__main__":    

    
    gen_images(50000, "data/charts", "data/charts/charts.h5")
    

    


