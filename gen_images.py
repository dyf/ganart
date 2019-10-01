import skimage.draw as skd
import skimage.io as skio
import numpy as np
import h5py
import itertools
import random

def gen_circle(shape, n_min, n_max):
    img = np.zeros(shape, dtype=float)

    bg = np.random.random(3)

    img[0,:,:] = bg[0]
    img[1,:,:] = bg[1]
    img[2,:,:] = bg[2]

    n = np.random.randint(n_min, n_max+1)

    r_max = min(shape[1]*0.3, shape[2]*0.3)
    r_min = 2    
    
    for i in range(n):
        color = np.random.random(3)

        radius = int(np.random.uniform(r_min, r_max))
        r = int(np.random.uniform(radius, shape[1]-radius))
        c = int(np.random.uniform(radius, shape[2]-radius))

        rr,cc = skd.circle(r, c, radius, shape=[shape[1], shape[2]])

        #rrs = np.array([rr] * len(dims)).flatten()
        #ccs = np.array([cc] * len(dims)).flatten()
        
        for ci,cv in enumerate(color):
            img[ci,rr,cc] = cv

    return img

def gen_circles(n, shape, n_min, n_max, fname):
    
    with h5py.File(fname, "w") as f:
        ds = f.create_dataset("data", (n,*shape), dtype='float32')

        for i in range(n):
            if i % 100 == 0:
                print(f'{i+1}/{n}')
            
            img = gen_circle(shape, n_min, n_max)
            ds[i,:] = img


if __name__ == "__main__":
    np.random.seed(0)

    gen_circles(20000, (3,256,256), 1, 15,
                "circles.h5")
        




