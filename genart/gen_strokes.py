import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

def place_strokes(shape, stroke_width):
    xx,yy = np.meshgrid(
        np.linspace(0.5, shape[0]-0.5, float(shape[0]) / stroke_width*2),
        np.linspace(0.5, shape[1]-0.5, float(shape[1]) / stroke_width*2)
    )
        
    stroke_positions = np.array(list(zip(xx.ravel(), yy.ravel())))
    
    jitter = np.random.normal(scale=stroke_width*0.5, size=(len(stroke_positions), 2))
    stroke_positions += jitter
    stroke_positions = np.round(stroke_positions).astype(int)
        
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
    return gxw.mean(axis=None), gyw.mean(axis=None)

def choose_stroke_color(img, pos, stroke_width):
    colors = image_window(img, pos, stroke_width)
    colors = colors.reshape(-1, img.shape[-1])
    
    # find unique colors
    unique_colors, counts = np.unique(colors, axis=0, return_counts=True)

    # find most common color
    mode_idx = np.argmax(counts)
    return unique_colors[mode_idx]

def stroke(ax, pos, w, color, gx, gy):    
    color = color / 255.0
    if gx == 0.0 and gy == 0.0:
        prim = mpatches.Circle((pos[1], pos[0]), radius=w*0.5, color=color)
    else:
        prim = mlines.Line2D((pos[1]-gx,pos[1]+gx),
                             (pos[0]-gy,pos[0]+gy),
                             linewidth=w*2,
                             solid_capstyle='round',
                             color=color)
    
    ax.add_artist(prim)
    

def stroke_image(img, stroke_width, stroke_length=None, curved=False):    
    stroke_positions = place_strokes(img.shape, stroke_width)

    gx, gy, gz = np.gradient(img)

    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(img)
    ax2.axis('square')
    ax2.set_xlim((0,img.shape[1]))
    ax2.set_ylim((0,img.shape[0]))
    
    
    for p in stroke_positions:
        color = choose_stroke_color(img, p, stroke_width)
        gxs, gys = image_gradient(gx, gy, p, stroke_width)
        stroke(ax2, p, stroke_width, color, gxs, gys)
    plt.show()
    return fig

if __name__ == "__main__":
    shape = (128,128)
    img = (np.random.random((shape[0],shape[1],3))*2).astype(int)*255
    xx,yy = np.meshgrid(np.linspace(0,1,shape[0]), np.linspace(0,1,shape[1]))
    mask = xx>0.25
    #img[mask,:] = 0
    #img[~mask,:] = 255
    

    print(img.shape)
    simg = stroke_image(img, 16)


