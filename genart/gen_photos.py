import tensorflow as tf
import numpy as np
from pathlib import Path
import re
import glob

def escape_path(p):
    return str(p).encode('unicode-escape').decode()

class IndexedImageLoader:
    def __init__(self, format):
        self.format = Path(format)

        fglob = str(self.format).format(index='*')
        
        self.image_files = [ Path(p) for p in glob.glob(fglob) ]

        self.idxs = []
        for im in self.image_files:
            toks = re.split('[.-]+', str(im))
            self.idxs.append(int(toks[-2]))

        self.idxs = sorted(self.idxs)

    def __len__(self):
        return len(self.idxs)

    def square_image(self, image):
        short_size = min(image.shape[0], image.shape[1])
        return image[:short_size, :short_size, :]

    def load_image(self, i=None, idx=None, square=True):
        if idx is None:
            if i is None:
                raise KeyError("not sure what I'm doing")

            idx = self.idxs[i]

        f = tf.io.read_file(str(self.format).format(index=idx))
        image = (tf.cast(tf.image.decode_jpeg(f), tf.float32) / 127.5) - 1.0
        
        if square:
            image = self.square_image(image)

        return image
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            imgs = []
            for i in range(*idx.indices(len(self))):
                imgs.append(self.load_image(i))                
            return tf.stack(imgs)
        else:
            return self.load_image(idx)[tf.newaxis,:,:,:]

    def load_patch(self, i, shape):
        im = self.load_image(i=i, square=True)

        start = np.array([ np.random.randint(0, im.shape[0] - shape[0]), 
                           np.random.randint(0, im.shape[1] - shape[1]) ])        

        return im[start[0]:start[0]+shape[0],
                  start[1]:start[1]+shape[1],:]
    
    def iter_patch(self, shape, batch_size):
        for i in range(0, len(self.idxs), batch_size):
            start_i = i
            end_i = min(i + batch_size, len(self.idxs))

            imgs = []
            for ii in range(start_i, end_i):
                imgs.append(self.load_patch(ii, shape))
            
            yield tf.stack(imgs)

class PairedImageLoader:
    def __init__(self, set_1_format, set_2_format):
        self.s1_loader = IndexedImageLoader(set_1_format)
        self.s2_loader = IndexedImageLoader(set_2_format)

        self.idxs = sorted(list(set(self.s1_loader.idxs) & set(self.s2_loader.idxs)))

        self.small_shape = (128,128,3)
        

    def __len__(self):
        return len(self.idxs)    
    
    def load_image_pair(self, i=None, idx=None, square=True):
        if idx is None:
            if i is None:
                raise KeyError("not sure what I'm doing")

            idx = self.idxs[i]

        return self.s1_loader.load_image(idx=idx, square=square), self.s2_loader.load_image(idx=idx, square=square)

    def load_patch_pair(self, i, shp1):
        im1, im2 = self.load_image_pair(i=i, square=True)

        sf = im2.shape[0] // im1.shape[0]
        shp2 = (shp1[0]*sf, shp1[1]*sf)

        start1 = np.array([ np.random.randint(0, im1.shape[0] - shp1[0]), np.random.randint(0, im1.shape[1] - shp1[1]) ])
        start2 = start1 * sf

        return (
            im1[start1[0]:start1[0]+shp1[0],
                start1[1]:start1[1]+shp1[1],:],
            im2[start2[0]:start2[0]+shp2[0],
                start2[1]:start2[1]+shp2[1],:]
        )
    
    def iter_patch_pair(self, shape, batch_size):
        for i in range(0, len(self.idxs), batch_size):
            start_i = i
            end_i = min(i + batch_size, len(self.idxs))

            imgs = []
            for ii in range(start_i, end_i):
                imgs.append(self.load_patch_pair(ii, shape))
            
            yield tf.stack([im[0] for im in imgs]), tf.stack([im[1] for im in imgs])
            
    
if __name__ == '__main__':
    imdl = PairedImageLoader('images/small/img-small-{index}.jpg', 'images/large/img-large-{index}.jpg')
    im1,im2 = imdl.load_image_pair_patch(0, (128,128))
    print(im1.shape, im2.shape)
    
    #print(imdl.small_images)




