import numpy as np

def stroke_image(img, stroke_width, stroke_length=None, curved=False):
    
    # compute stroke positions
    N = int(img.shape[0] * img.shape[1] / stroke_width**2)    
    
    stroke_positions = np.random.random((N,2))
    stroke_positions[:,0] *= img.shape[0]
    stroke_positions[:,1] *= img.shape[1]

    # compute colors at positions
    xx,yy = np.meshgrid(np.arange(0,img.shape[0]), np.arange(0,img.shape[1]))

    stroke_colors = np.zeros((stroke_positions.shape[0], stroke_positions.shape[1], img.shape[2]), dtype=float)
    hw = stroke_width*0.5
    for i in range(N):
        # find colors
        p = stroke_positions[i]
        cpos = np.where((xx>p[0]-hw)&(xx<p[0]+hw)&(yy>p[1]-hw)&(yy<p[1]+hw))  
        colors = img[cpos]

        # find unique colors
        unique_colors, counts = np.unique(colors, axis=0, return_counts=True)

        # find most common color
        mode_idx = np.argmax(counts)
        stroke_colors[i] = unique_colors[mode_idx]        

    # apply strokes
    for i in range(N):
        pass
    
    return None
if __name__ == "__main__":
    shape = (128,128)
    img = (np.random.random((shape[0],shape[1],3))*2).astype(int)
    xx,yy = np.meshgrid(np.linspace(0,1,shape[0]), np.linspace(0,1,shape[1]))
    mask = xx>0.5
    img[mask,:] = 0

    print(img.shape)
    simg = stroke_image(img, 16)


