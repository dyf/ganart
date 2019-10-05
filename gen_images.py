import skimage.draw as skd
import skimage.io as skio
import numpy as np
import h5py
import itertools
import random

def gen_circle(shape, n_min, n_max, dtype=np.float32):
    img = np.zeros(shape, dtype=dtype)

    bg = np.random.random(3)

    img[0,:,:] = bg[0]
    img[1,:,:] = bg[1]
    img[2,:,:] = bg[2]

    n = np.random.randint(n_min, n_max+1)

    r_max = min(shape[1]*0.75, shape[2]*0.75)
    r_min = 2    
    radii = np.random.uniform(r_min, r_max, n).astype(int)
    radii = np.sort(radii)[::-1]    
    
    for i in range(n):
        color = np.random.random(3)

        radius = radii[i]
        r = int(np.random.uniform(0, shape[1]-1))
        c = int(np.random.uniform(0, shape[2]-1))

        rr,cc = skd.circle(r, c, radius, shape=[shape[1], shape[2]])

        #rrs = np.array([rr] * len(dims)).flatten()
        #ccs = np.array([cc] * len(dims)).flatten()
        
        for ci,cv in enumerate(color):
            img[ci,rr,cc] = cv

    return img

def gen_circles(n, shape, n_min, n_max, fname, dtype=np.float32):
    
    with h5py.File(fname, "w") as f:
        ds = f.create_dataset("data", (n,*shape), dtype=dtype)

        for i in range(n):
            if i % 100 == 0:
                print(f'{i+1}/{n}')
            
            img = gen_circle(shape, n_min, n_max, dtype)
            ds[i,:] = img


if __name__ == "__main__":
    np.random.seed(0)

    gen_circles(20000, (3,256,256), 1, 20,
                "circles.h5", dtype=np.float32)
        




