import skimage.draw as skd
import skimage.io as skio
import numpy as np
import h5py
import itertools
import random
from typing import List
from dataclasses import dataclass, field

def default_float(n=1,low=0.0,high=1.0):
    if n == 1:
        return field(default_factory = lambda: np.random.uniform(low, high) )
    else:
        return field(default_factory = lambda: np.random.uniform(low, high, n) )

@dataclass
class Shape:
    color: List[float] = default_float(3)
    size: float = default_float(low=.1,high=.9)
    x: float = default_float()
    y: float = default_float()

    def gen(self, img_size):
        raise NotImplementedError()

@dataclass
class RotatableShape(Shape):
    rotation: float = default_float()

class Circle(Shape): 
    def render(self, img_size):
        radius = int(self.size * 0.5 * min(img_size[0], img_size[1]))
        r = int(self.y*img_size[0])
        c = int(self.x*img_size[1])
        return skd.circle(r, c, radius, shape=img_size[:2])

class Square(RotatableShape):
    def render(self, img_size):
        r = int(self.y*img_size[0])
        c = int(self.x*img_size[1])
        th = self.rotation * np.pi

        hs = int(self.size * 0.5 * min(img_size[0], img_size[1]))        

        rr = [r-hs, r-hs, r+hs, r+hs]
        cc = [c-hs, c+hs, c+hs, c-hs]
        
        rr, cc = rotate_rc(rr, cc, th, r, c)        

        rr,cc = skd.polygon(rr, cc)
        rr,cc = rr.flatten(), cc.flatten()

        return rr, cc

@dataclass
class Rectangle(RotatableShape):
    aspect: float = default_float(low=0.0,high=1.0)

    def render(self, img_size):
        r = int(self.y * img_size[0])
        c = int(self.x * img_size[1])
        th = self.rotation * np.pi

        hs = int(self.size * 0.5 * min(img_size[0], img_size[1]))        

        rr = [r-hs, r-hs, r+hs, r+hs]
        cc = [c-hs*self.aspect, c+hs*self.aspect, c+hs*self.aspect, c-hs*self.aspect]

        rr, cc = rotate_rc(rr, cc, th, r, c)

        rr,cc = skd.polygon(rr, cc)
        rr,cc = rr.flatten(), cc.flatten()

        return rr, cc

@dataclass
class Ellipse(RotatableShape):
    aspect: float = default_float(low=0.0,high=1.0)

    def render(self, img_size):
        r = int(self.y * img_size[0])
        c = int(self.x * img_size[1])
        th = self.rotation * 2 * np.pi - np.pi

        radius = int(self.size * 0.5 * min(img_size[0], img_size[1]))       
        
        rr,cc = skd.ellipse(r, c, radius, radius*self.aspect, rotation=th)
        rr,cc = rr.flatten(), cc.flatten()
        return rr,cc

class Triangle(RotatableShape):
    def render(self, img_size):
        r = int(self.y * img_size[0])
        c = int(self.x * img_size[1])
        th = self.rotation * 2 * np.pi

        hw = int(self.size * 0.5 * min(img_size[0], img_size[1]))       
        hh = hw*np.sqrt(3) * 0.5

        rr = [ r+hw, r, r-hw ]
        cc = [ c-hh, c+hh, c-hh ]

        rr, cc = rotate_rc(rr, cc, th, r, c)

        rr,cc = skd.polygon(rr, cc)
        rr,cc = rr.flatten(), cc.flatten()

        return rr, cc

SHAPE_CHOICES = [ Circle, Triangle, Rectangle, Ellipse, Square ]

def render_shapes(shapes, img_size):    
    img = np.zeros(img_size, dtype=np.float32)
    
    bg = np.random.random(3)
    
    img[:,:,0] = bg[0]
    img[:,:,1] = bg[1]
    img[:,:,2] = bg[2]

    for shape in shapes:
        rr,cc = shape.render(img_size)        
        rr,cc = crop_rc(rr, cc, img_size)

        img[rr,cc,:] = shape.color
    
    return img

def random_shapes(shape, n_min, n_max):            
    n = np.random.randint(n_min, n_max+1)

    shapes = [ random_shape(shape) for i in range(n) ]
    shapes.sort(key=lambda s: s.size)
    shapes = shapes[::-1]

    return shapes

def random_shape(shape=None):
    if shape is None:
        shape = random.choice(SHAPE_CHOICES)
    return shape()    
    

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

def crop_rc(rr, cc, img_size):
    mask = (rr >= 0) & (rr < img_size[0]) & (cc >= 0) & (cc < img_size[1])     
    return rr[mask], cc[mask]

def render_shape_sets(n, shape, img_sizes, n_min, n_max, dtype=np.float32):
    img_sets = [ np.zeros([n]+list(img_size), dtype=dtype) for img_size in img_sizes ]

    for i in range(n):
        shapes = random_shapes(shape, n_min, n_max)

        for j in range(len(img_sizes)):            
            img_sets[j][i,:] = render_shapes(shapes, img_sizes[j])

    return img_sets

           


if __name__ == "__main__":
    shp = (512,512,3)
    im = render_shape_sets(1, None, [shp,shp], 10, 10)
    
    import matplotlib.pyplot as plt
    plt.imshow(im[1][0])
    plt.axis('off')
    plt.show()    
    plt.savefig('test.png')

        




