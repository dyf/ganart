import skimage.draw as skd
import skimage.io as skio
import numpy as np
import h5py
import itertools
import random

def gen_circle(shape, n_min, n_max):
    img = np.zeros(shape, dtype=float)

    n = np.random.randint(n_min, n_max+1)

    r_max = min(shape[1]*0.3, shape[2]*0.3)
    r_min = 2

    dim_combs = [ list(itertools.combinations(np.arange(shape[0]),i)) for i in range(1,shape[0]+1) ]
    dim_combs_flat = []
    for dc in dim_combs:
        dim_combs_flat += dc
        
    for i in range(n):
        radius = int(np.random.uniform(r_min, r_max))
        r = int(np.random.uniform(radius, shape[1]-radius))
        c = int(np.random.uniform(radius, shape[2]-radius))
        dims = np.array(random.choice(dim_combs_flat))

        rr,cc = skd.circle(r, c, radius, shape=[shape[1], shape[2]])

        npx = len(rr)

        rrs = np.array([rr] * len(dims)).flatten()
        ccs = np.array([cc] * len(dims)).flatten()
        dds = np.repeat(dims, len(rr))

        img[dds,rrs,ccs] += 1

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

    gen_circles(10000, (3,256,256), 1, 10,
                "circles.h5")
        




