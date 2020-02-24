import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import imageio
import copy 
import scipy.ndimage
import scipy.signal

import skimage.feature as skf
import skimage.transform as skt

from dataclasses import dataclass
from typing import Tuple

DPI = 1.0

@dataclass 
class Shape:
    color: Tuple[float, float, float]

    def _defcol(self, color):
        return self.color if color is None else color    

@dataclass
class Circle(Shape):
    pos: Tuple[float, float]
    radius: float

    def make_artists(self, color=None):
        return [
            mpatches.Circle(xy=self.pos, radius=self.radius, color=self._defcol(color))
        ]
    
    def make_skeleton_artists(self, color=None):
        return [
            mpatches.Circle(xy=self.pos, radius=0.5, color=self._defcol(color), antialiased=False)
        ]

    def scaled(self, s):
        return Circle(pos=(np.array(self.pos)*s).astype(int), radius=self.radius*s, color=self.color)


@dataclass
class Line(Shape):
    p0: Tuple[float, float] = (0.0, 0.0)
    p1: Tuple[float, float] = (0.0, 0.0)
    width: float = 0.0
    v_tan: Tuple[float, float] = (0.0, 0.0)

    def scaled(self, s):
        return Line(p0=(np.array(self.p0)*s).astype(int),
                    p1=(np.array(self.p1)*s).astype(int),
                    width=self.width*s, v_tan=self.v_tan, color=self.color)

    def __post_init__(self):
        dp = np.array(self.p1) - np.array(self.p0)
        self.v_tan = np.array([ -dp[1], dp[0] ]).astype(float)
        self.v_tan /= np.linalg.norm(self.v_tan)

    def compute_rect(self):
        hw = self.width*0.5

        return [ 
            self.p0 + self.v_tan*hw,
            self.p0 - self.v_tan*hw,
            self.p1 - self.v_tan*hw,
            self.p1 + self.v_tan*hw 
        ]

    def make_artists(self, color=None):
        hw = self.width*0.5

        return [
            mpatches.Polygon(self.compute_rect(),
                             closed=True,
                             facecolor=self._defcol(color),
                             edgecolor=None),
            mpatches.Circle(self.p0, radius=hw, facecolor=self._defcol(color), edgecolor=None),
            mpatches.Circle(self.p1, radius=hw, facecolor=self._defcol(color), edgecolor=None)
        ]
    
    def make_skeleton_artists(self, color):
        return [
            mlines.Line2D([ self.p0[0], self.p1[0] ], 
                          [ self.p0[1], self.p1[1] ],
                          linewidth=1,
                          color=color,
                          antialiased=False)
        ]

@dataclass
class GradientLine(Line):
    pos: Tuple[float, float] = (0.0, 0.0)
    gradient: Tuple[float, float] = (0.0, 0.0)
    
    def scaled(self, s):
        return GradientLine(pos=(np.array(self.pos)*s).astype(int),
                            gradient=self.gradient,
                            p0=(np.array(self.p0)*s).astype(int),
                            p1=(np.array(self.p1)*s).astype(int),  
                            width=self.width*s, v_tan=self.v_tan, color=self.color)

    def __post_init__(self):
        length = np.linalg.norm(self.gradient)
        self.v_tan = np.array(self.gradient) / length

        v_dir = np.array([ -self.v_tan[1], self.v_tan[0] ])
        v_dir /= np.linalg.norm(v_dir)

        self.p0 = self.pos - v_dir*length*0.5 
        self.p1 = self.pos + v_dir*length*0.5 

    

class Renderer:
    def draw_skeleton(self, artists):
        pass

    def draw(self, artists):
        pass

def detect_edges(img):
    edges = [] 
    for i in range(img.shape[2]):
        edges.append(skf.canny(img[:,:,i], 2, 1, 25))
    
    
    edges = np.array(edges).max(axis=0)

    return skt.probabilistic_hough_line(edges, threshold=10, line_length=10, line_gap=3)
    
def place_strokes(shape, stroke_width):
    xx,yy = np.meshgrid(
        np.linspace(0.5, shape[0]-0.5, int(shape[0] / stroke_width*2 / 0.75)),
        np.linspace(0.5, shape[1]-0.5, int(shape[1] / stroke_width*2 / 0.75))
    )
        
    stroke_positions = np.array(list(zip(xx.ravel(), yy.ravel())))
    
    jitter = np.random.normal(scale=stroke_width*0.5, size=(len(stroke_positions), 2))
    stroke_positions += jitter
    stroke_positions = np.round(stroke_positions).astype(int)
    np.random.shuffle(stroke_positions)
        
    return stroke_positions[
        (stroke_positions[:,0] >= 0) & 
        (stroke_positions[:,1] >= 0) & 
        (stroke_positions[:,0] < shape[0]) & 
        (stroke_positions[:,1] < shape[1])
    ]
    

def image_window(img, pos, stroke_width):
    hw = stroke_width*0.5
    LL = np.round([pos[0]-hw,pos[1]-hw]).astype(int)
    UR = LL + stroke_width
    
    LL = [ max(LL[0], 0), max(LL[1], 0) ]
    UR = [ min(UR[0], img.shape[0]), min(UR[1], img.shape[1]) ]    

    return img[LL[0]:UR[0], LL[1]:UR[1]]

def image_gradient(gx, gy, pos, stroke_width):    
    gxw, gyw = image_window(gx, pos, stroke_width), image_window(gy, pos, stroke_width)    
    return gxw.mean(axis=(0,1)), gyw.mean(axis=(0,1))

def image_orientation(gx, gy, pos, stroke_width):
    gxm, gym = image_gradient(gx, gy, pos, stroke_width)

    a = np.dot(gxm, gxm)
    b = np.dot(gym, gym)
    c = np.dot(gxm, gym)

    #lam1 = a + b + np.sqrt((a-b)**2 + 4*a*c)*0.5
    #lam2 = a + b - np.sqrt((a-b)**2 + 4*a*c)*0.5

    return a - b + np.sqrt((a-b)**2 +4*c*c), 2*c


def choose_stroke_color(img, pos, stroke_width):
    colors = image_window(img, pos, stroke_width)
    colors = colors.reshape(-1, img.shape[-1])
    
    # find unique colors
    unique_colors, counts = np.unique(colors, axis=0, return_counts=True)

    # find most common color
    mode_idx = np.argmax(counts)
    return unique_colors[mode_idx]

def make_stroke(pos, w, color, ox, oy): 

    # rendering is transposed from indexing
    pos = [pos[1], pos[0]]    

    shapes = []

    if ox == 0.0 and oy == 0.0:
        circle = Circle(pos=pos, radius=w*0.5, color=color)
        shapes.append(circle)
    else:
        line = GradientLine(pos=pos, gradient=[ oy, ox ], width=w, color=color)
        shapes.append(line)

    return shapes                             

def stroke_height_simple(shapes, img_shape, max_height):   
    artists = [ a for s in shapes for a in s.make_skeleton_artists(color='black') ]
    img = render_artists(artists, img_shape)

    dist = scipy.ndimage.distance_transform_edt(img[:,:,0]>0)
    dist[dist > max_height] = 0
    dist /= max_height

    return np.power(dist, 1.5)

def stroke_height_full(shapes, img_shape, max_height):
    height_map = np.zeros((img_shape[0], img_shape[1]), dtype=float)
    for i,shape in enumerate(shapes):
        if i % 10 == 0:
            print(f"{i+1}/{len(shapes)}")
        img = render_artists(shape.make_artists(color='black'), img_shape)

        img = img[:,:,0] == 0

        img = scipy.ndimage.distance_transform_edt(img)
        
        img_max = img.max()
        inz = img > 0
        img[inz] = (1.0 - img[inz]/img_max)
        img = np.power(img, 1.5) * np.random.normal(loc=1.0, scale=0.2)
        
        height_map[inz] = img[inz]
    
    return height_map



def emboss(img, dir='above', k=2):
    xx,yy = np.meshgrid(np.arange(-k,k+1), np.arange(-k,k+1))

    kernel = np.zeros(xx.shape, dtype=int)
    if dir == 'above':
        kernel[(yy < 0) & (xx == 0)] = 1
        kernel[(yy > 0) & (xx == 0)] = -1
    else:
        raise Exception(f"direction unknown: {dir}")

    outim = scipy.signal.convolve2d(img, kernel, boundary='symm')
    return outim[k:-k,k:-k]

def render_artists(artists, shape):
    fig = plt.figure(figsize=(shape[1],shape[0]), dpi=DPI)
    ax = plt.axes([0,0,1,1])

    for artist in artists:
        artist = copy.copy(artist)
        ax.add_artist(artist)

    
    ax.axis('square')
    ax.axis('off')
    ax.set_xlim((0,shape[1]))
    ax.set_ylim((0,shape[0]))
    ax.invert_yaxis()

    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())    
    plt.close(fig)

    return img

def stroke_image(img, stroke_width, stroke_length=None, curved=False, gscale=1.0, out_width=None):
    if out_width is None:
        out_width = img.shape[1]            
    
    scale_factor = float(out_width) / img.shape[1]
    out_shape = ( int(scale_factor*img.shape[0]), int(scale_factor*img.shape[1]) )    
    out_stroke_width = stroke_width * scale_factor

    # place the strokes in image coordinates
    stroke_positions = place_strokes(img.shape, stroke_width)
    stroke_widths = np.random.normal(loc=stroke_width, scale=.15 * stroke_width, size=stroke_positions.shape[0]).astype(int)
    
    gx, gy, gz = np.gradient(img)    

    # build simple strokes        
    all_shapes = []
    N = len(stroke_positions)
    for i in range(N):
        p = stroke_positions[i]
        w = stroke_widths[i]

        color = choose_stroke_color(img, p, w) / 255.0
        ox, oy = image_orientation(gx, gy, p, w)  
        shapes = make_stroke(p, w, color, ox*gscale, oy*gscale)

        all_shapes += shapes

    # build additional edge strokes
    edges = detect_edges(img)
    for p0, p1 in edges:
        p = (0.5 * (np.array(p0) + np.array(p1))).astype(int)
        p = np.array([p[1], p[0]])
        w = int(np.random.normal(loc=stroke_width, scale=0.15*stroke_width))
        color = choose_stroke_color(img, p, w)
        all_shapes.append(Line(p0=p0, p1=p1, 
                               color=color / 255.0, 
                               width=w))

    # build the embossed height map
    scaled_shapes = [ s.scaled(scale_factor) for s in all_shapes ]    
    height_im = stroke_height_full(scaled_shapes, out_shape, stroke_width*0.5*scale_factor)
    
    r = np.abs(np.random.normal(scale=0.3, size=height_im.shape))
    height_im = emboss(height_im*30 + r)   
    
    all_artists = [ a for s in scaled_shapes for a in s.make_artists() ]
    line_im = render_artists(all_artists, out_shape)[:,:,:3]
    composite = np.clip(line_im.astype(float) + np.dstack([height_im, height_im, height_im]), 0, 255).astype(np.uint8)

    return composite

if __name__ == "__main__":
    #DPI = 0.5
    DPI = 1.0
    shape = (128,128)
    img = (np.random.random((shape[0],shape[1],3))*2).astype(int)*255
    img.fill(0)
    xx,yy = np.meshgrid(np.linspace(0,1,shape[0]), np.linspace(0,1,shape[1]))
    
    mask = (xx-0.25)**2 + (yy-0.75)**2 < 0.05
    #mask = xx < yy
    img[mask,0] = 255
    
    mask = xx >= yy
    img[mask,1] = 255

    mask = (xx-0.5)**2 + (yy-0.5)**2 < 0.05
    img[mask,2] = 255

    img = imageio.imread("octopus.jpg")[::2,::2,:3].astype(int)
    
    simg = stroke_image(img, 6, gscale=0.05 , out_width=2000)

    imageio.imwrite('test.png', simg)


