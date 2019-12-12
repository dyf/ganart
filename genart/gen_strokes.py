import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import imageio
import copy 
import scipy.ndimage
import scipy.signal

from dataclasses import dataclass
from typing import Tuple

DPI = 1.0

@dataclass
class Circle:
    pos: Tuple[float, float]
    radius: float
    color: Tuple[float, float, float]

    def make_artists(self):
        return [
            mpatches.Circle(xy=self.pos, radius=self.radius, color=self.color)
        ]
    
    def make_skeleton_artists(self, color):
        return [
            mpatches.Circle(xy=self.pos, radius=0.5, color=color, antialiased=False)
        ]

@dataclass
class OrientedLine:
    pos: Tuple[float, float]
    ori: Tuple[float, float]
    width: float
    length: float
    color: Tuple[float, float, float]

    def compute_endpoints(self):
        return [ self.pos - self.ori*self.length*0.5,
                 self.pos + self.ori*self.length*0.5 ]

    def compute_rect(self, p0, p1):       
        or_t = [-self.ori[1], self.ori[0]]
        or_t /= np.linalg.norm(or_t)

        hw = self.width*0.5

        return [ 
            p0 + or_t*hw,
            p0 - or_t*hw,
            p1 - or_t*hw,
            p1 + or_t*hw 
        ]

    def make_artists(self):
        p0, p1 = self.compute_endpoints()
        rect = self.compute_rect(p0, p1)
        
        hw = self.width*0.5

        return [
            mpatches.Polygon(rect,
                             closed=True,
                             facecolor=self.color,
                             edgecolor=None),
            mpatches.Circle(p0, radius=hw, facecolor=self.color, edgecolor=None),
            mpatches.Circle(p1, radius=hw, facecolor=self.color, edgecolor=None)
        ]
    
    def make_skeleton_artists(self, color):
        p0, p1 = self.compute_endpoints()
        return [
            mlines.Line2D([ p0[0], p1[0] ], 
                          [ p0[1], p1[1] ],
                          linewidth=1,
                          color=color,
                          antialiased=False)
        ]

class Renderer:
    def draw_skeleton(self, artists):
        pass

    def draw(self, artists):
        pass

    
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
    color = color / 255.0

    # rendering is transposed from indexing
    pos = [pos[1], pos[0]]    

    shapes = []

    if ox == 0.0 and oy == 0.0:
        circle = Circle(pos=pos, radius=w*0.5, color=color)
        shapes.append(circle)
    else:
        or_g = [ oy, ox ]
        gmag = np.linalg.norm(or_g)
        or_g /= gmag

        or_t = [ -or_g[1], or_g[0] ]
        or_t /= np.linalg.norm(or_t)

        line = OrientedLine(pos=pos, ori=or_t, width=w, length=gmag, color=color)
        shapes.append(line)

    return shapes                             

def stroke_height(shapes, img_shape, max_height):   
    artists = [ a for s in shapes for a in s.make_skeleton_artists('black') ]
    img = render_artists(artists, img_shape)

    dist = scipy.ndimage.distance_transform_edt(img[:,:,0]>0)
    dist[dist > max_height] = 0
    dist /= max_height

    return np.power(dist, 1.5)

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
        
    fig, (ax1,ax2,ax3) = plt.subplots(1, 3)
    ax1.imshow(img)
    ax2.axis('square')
    ax2.set_xlim((0,img.shape[1]))
    ax2.set_ylim((0,img.shape[0]))
    ax2.invert_yaxis()
    ax3.invert_yaxis()

    gx, gy, gz = np.gradient(img)    
    
    scale_factor = float(out_width) / img.shape[1]
    out_shape = ( int(scale_factor*img.shape[0]), int(scale_factor*img.shape[1]) )    
    out_stroke_width = stroke_width * scale_factor

    stroke_positions = place_strokes(img.shape, stroke_width)
        
    all_shapes = []
    for p in stroke_positions:
        color = choose_stroke_color(img, p, stroke_width)
        ox, oy = image_orientation(gx, gy, p, stroke_width)        
        shapes = make_stroke(p, stroke_width, color, ox*gscale, oy*gscale)

        all_shapes += shapes

    height_im = stroke_height(all_shapes, img.shape, stroke_width*0.5)
    
    r = np.abs(np.random.normal(scale=0.3, size=height_im.shape))
    height_im = emboss(height_im*30 + r)   
    ax2.imshow(height_im, cmap='gray')
    
    all_artists = [ a for s in all_shapes for a in s.make_artists() ]
    line_im = render_artists(all_artists, img.shape)[:,:,:3]
    composite = np.clip(line_im.astype(float) + np.dstack([height_im, height_im, height_im]), 0, 255).astype(np.uint8)
    ax3.imshow(composite)#, alpha=0.9)
    plt.show()
    return fig

if __name__ == "__main__":
    DPI = 0.5
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

    

    img = imageio.imread("octopus.png")[::3,::3,:3].astype(int)
    

    print(img.shape)
    simg = stroke_image(img, 10, gscale=0.05 , out_width=500)


