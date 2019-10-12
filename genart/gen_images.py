import skimage.draw as skd
import skimage.io as skio
import numpy as np
import h5py
import itertools
import random

def gen_shapes(shape, img_shape, n_min, n_max):
    img = np.zeros(img_shape, dtype=np.float32)
    
    bg = np.random.random(3)

    img[:,:,0] = bg[0]
    img[:,:,1] = bg[1]
    img[:,:,2] = bg[2]

    n = np.random.randint(n_min, n_max+1)

    size_max = min(img_shape[0]*0.75, img_shape[1]*0.75)
    size_min = 2

    sizes = np.random.uniform(size_min, size_max, n).astype(int)
    sizes = np.sort(sizes)[::-1]

    for size in sizes:
        color = np.random.random(3)

        rr,cc = random_shape(shape, img_shape[:2], size)
        img[rr,cc,:] = color

    return img



def random_shape(shape, img_shape, size):
    if shape is None:
        shape = random.choice(['circle', 'square', 'rectangle', 'ellipse', 'triangle'])

    if shape == 'circle':
        out = random_circle(img_shape, size)
    elif shape == 'square':
        out = random_square(img_shape, size)
    elif shape == 'rectangle':
        out = random_rectangle(img_shape, size)
    elif shape == 'ellipse':
        out =random_ellipse(img_shape, size)
    elif shape == 'triangle':
        out = random_triangle(img_shape, size)
    else:
        raise KeyError(f"Unknown shape type: {shape}")

    return crop_rc(*out, img_shape)

def rotate_rc(rr,cc,th,r,c):
    p = np.array([ rr, 
                   cc, 
                   np.ones(len(rr)) ])

    T1 = np.array([[ 1, 0, -r],
                   [ 0, 1, -c],
                   [ 0, 0, 1 ]])

    R = np.array([ [ np.cos(th), -np.sin(th), 0 ],
                   [ np.sin(th), np.cos(th),  0 ],
                   [ 0,          0,           1 ] ])

    T2 = np.array([[ 1, 0, r ],
                   [ 0, 1, c ],
                   [ 0, 0, 1 ]])

    T = np.dot(T2, np.dot(R, T1))

    pt = np.dot(T, p)
    pt = np.round(pt)

    return pt[0].astype(int), pt[1].astype(int)

def crop_rc(rr, cc, img_shape):
    mask = (rr >= 0) & (rr < img_shape[0]) & (cc >= 0) & (cc < img_shape[1])     
    return rr[mask], cc[mask]

def random_circle(img_shape, size):
    if size is None:
        pass # todo

    radius = size * 0.5
    r = int(np.random.uniform(0, img_shape[0]-1))
    c = int(np.random.uniform(0, img_shape[1]-1))

    return skd.circle(r, c, radius, shape=[img_shape[0], img_shape[1]])

def random_square(img_shape, size):
    r = int(np.random.uniform(0, img_shape[0]-1))
    c = int(np.random.uniform(0, img_shape[1]-1))
    th = np.random.uniform(0,2*np.pi)

    hs = size // 2

    rr = [r-hs, r-hs, r+hs, r+hs]
    cc = [c-hs, c+hs, c+hs, c-hs]
    rr, cc = rotate_rc(rr, cc, th, r, c)

    rr,cc = skd.polygon(rr, cc)
    rr,cc = rr.flatten(), cc.flatten()

    return rr, cc

def random_rectangle(img_shape, size):
    r = int(np.random.uniform(0, img_shape[0]-1))
    c = int(np.random.uniform(0, img_shape[1]-1))
    f = np.random.uniform(0.05,1)    
    th = np.random.uniform(0,2*np.pi)

    rr = [r-size, r-size, r+size, r+size]
    cc = [c-size*f, c+size*f, c+size*f, c-size*f]

    rr, cc = rotate_rc(rr, cc, th, r, c)

    rr,cc = skd.polygon(rr, cc)
    rr,cc = rr.flatten(), cc.flatten()

    return rr, cc

def random_ellipse(img_shape, size):
    r = int(np.random.uniform(0, img_shape[0]-1))
    c = int(np.random.uniform(0, img_shape[1]-1))
    f = np.random.uniform(0.05,1)    
    th = np.random.uniform(-np.pi, np.pi)

    rr,cc = skd.ellipse(r, c, size, size*f, rotation=th)
    rr,cc = rr.flatten(), cc.flatten()
    return rr,cc

def random_triangle(img_shape, size):
    r = int(np.random.uniform(0, img_shape[0]-1))
    c = int(np.random.uniform(0, img_shape[1]-1))    
    th = np.random.uniform(-np.pi, np.pi)

    hw = size * 0.5
    hh = hw*np.sqrt(3) * 0.5

    rr = [ r+hw, r, r-hw ]
    cc = [ c-hh, c+hh, c-hh ]

    rr, cc = rotate_rc(rr, cc, th, r, c)

    rr,cc = skd.polygon(rr, cc)
    rr,cc = rr.flatten(), cc.flatten()

    return rr, cc

def gen_shapes_set(n, shape, img_shape, n_min, n_max, dtype=np.float32):
    ds = np.zeros([n]+list(img_shape), dtype=dtype)

    for i in range(n):
        ds[i,:] = gen_shapes(shape, img_shape, n_min, n_max)

    return ds

           


if __name__ == "__main__":
    #np.random.seed(0)

    shp = (100,100,3)
    im = gen_shapes_set(8, None, shp, 1, 10)
    print(im.shape)
    import matplotlib.pyplot as plt
    plt.imshow(im[0])
    plt.savefig('foo.png')    


        




