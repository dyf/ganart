import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import imageio
import copy 
import scipy.ndimage
import scipy.signal


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

def stroke(pos, w, color, ox, oy):
    color = color / 255.0

    # rendering is transposed from indexing
    pos = [pos[1], pos[0]]    

    artists = []

    if ox == 0.0 and oy == 0.0:
        artists += [
            mpatches.Circle(pos, radius=w*0.5, color=color)
        ]
    else:
        or_g = [oy, ox]
        gmag = np.linalg.norm(or_g)
        or_g /= gmag

        or_t = [-or_g[1], or_g[0]]
        or_t /= np.linalg.norm(or_t)

        p0 = pos - or_t*gmag
        p1 = pos + or_t*gmag

        artists += [
            mpatches.Polygon([ p0 + or_g*w*0.5, 
                               p0 - or_g*w*0.5,
                               p1 - or_g*w*0.5,
                               p1 + or_g*w*0.5],
                             closed=True,
                             facecolor=color,
                             edgecolor=None),
            mpatches.Circle(p0, radius=w*0.5, facecolor=color, edgecolor=None),
            mpatches.Circle(p1, radius=w*0.5, facecolor=color, edgecolor=None)
        ]

    return artists                             

def stroke_height(artists, shape, dpi=100):   
    fig = plt.figure(figsize=(shape[0]/dpi,shape[1]/dpi))
    ax = plt.axes([0,0,1,1])

    for artist in artists:
        artist = copy.copy(artist)
        artist.set_facecolor('black')
        ax.add_artist(artist)

    
    ax.axis('square')
    ax.axis('off')
    ax.set_xlim((0,shape[1]))
    ax.set_ylim((0,shape[0]))
    ax.invert_yaxis()

    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())    
    plt.close(fig)
    
    dist = scipy.ndimage.distance_transform_edt(img[:,:,0]==0)
    dmax = dist.max()
    dnz = dist > 0
    dist[dnz] = np.power(dmax - dist[dnz], 0.8) + 1
    return dist

def emboss(img, dir='above', k=2):
    xx,yy = np.meshgrid(np.arange(-k,k+1), np.arange(-k,k+1))

    kernel = np.zeros(xx.shape, dtype=int)
    if dir == 'above':
        kernel[(yy < 0) & (xx == 0)] = 1
        kernel[(yy > 0) & (xx == 0)] = -1
    else:
        raise Exception(f"direction unknown: {dir}")

    return scipy.signal.convolve2d(img, kernel, boundary='symm')


def stroke_image(img, stroke_width, stroke_length=None, curved=False, gscale=1.0):
    stroke_positions = place_strokes(img.shape, stroke_width)

    gx, gy, gz = np.gradient(img)    

    fig, (ax1,ax2,ax3) = plt.subplots(1, 3)
    ax1.imshow(img)
    ax2.axis('square')
    ax2.set_xlim((0,img.shape[1]))
    ax2.set_ylim((0,img.shape[0]))
    ax2.invert_yaxis()

    ax3.invert_yaxis()

    height_im = np.zeros((img.shape[0], img.shape[1]), dtype=float)
        
    for p in stroke_positions:
        color = choose_stroke_color(img, p, stroke_width)
        ox, oy = image_orientation(gx, gy, p, stroke_width)        
        artists = stroke(p, stroke_width, color, ox*gscale, oy*gscale)
        
        ph = stroke_height(artists, img.shape)
        
        height_im[ph>0] = ph[ph>0]

        for artist in artists:
            ax2.add_artist(artist)        

    
    r = np.abs(np.random.normal(scale=0.5, size=height_im.shape))
    height_im = emboss(height_im + r) 
    
    ax3.imshow(height_im, cmap='gray')
    plt.show()

    ax2.imshow(height_im*255, cmap='gray')#, alpha=0.9)
    return fig

if __name__ == "__main__":
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

    

    img = imageio.imread("octopus5.png")[:,:,:3].astype(int)
    

    print(img.shape)
    simg = stroke_image(img, 20, gscale=0.05)


