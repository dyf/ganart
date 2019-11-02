import tensorflow as tf
import numpy as np
from pathlib import Path
import re

class ScaledImageDataLoader:
    def __init__(self, basedir='./images'):
        self.basedir = Path(basedir)

        self.small_images = list(self.basedir.glob("small/img-small*"))
        self.large_images = list(self.basedir.glob("large/img-large*"))

        self.small_shape = (128,128,3)
        
        small_idxs = set()
        for im in self.small_images:
            toks = re.split('[\.-]',str(im))
            small_idxs.add(int(toks[-2]))

        large_idxs = set()
        for im in self.small_images:
            toks = re.split('[\.-]',str(im))
            large_idxs.add(int(toks[-2]))
        
        self.all_idxs = sorted(list(small_idxs & large_idxs))    

    def __len__(self):
        return len(self.all_idxs)

    def square_image(self, image):
        short_size = min(image.shape[0], image.shape[1])
        return image[:short_size, :short_size, :]

    def load_image(self, i):
        idx = self.all_idxs[i]

        f = tf.io.read_file(str(self.basedir / 'small' / f'img-small-{idx}.jpg'))
        small_image = (tf.cast(tf.image.decode_jpeg(f), tf.float32) / 127.5) - 1.0
        f = tf.io.read_file(str(self.basedir / 'large' / f'img-large-{idx}.jpg'))
        large_image = (tf.cast(tf.image.decode_jpeg(f), tf.float32) / 127.5) - 1.0

        small_image = self.square_image(small_image)
        large_image = self.square_image(large_image)

        if small_image.shape[0] != 200:
            raise ValueError(f'image {i} has a weird shape: {str(small_image.shape)}')
        elif large_image.shape[0] != 800:
            raise ValueError(f'image {i} has a weird shape: {str(large_image.shape)}')

        sf = large_image.shape[0] // small_image.shape[0]
        
        start_sm = np.random.randint(0, small_image.shape[0] - self.small_shape[0], 2)
        start_lg = start_sm * sf
        large_shape = (self.small_shape[0]*sf, self.small_shape[1]*sf, 3)
        return (
            small_image[start_sm[0]:start_sm[0]+self.small_shape[0],
                        start_sm[1]:start_sm[1]+self.small_shape[1],:],
            large_image[start_lg[0]:start_lg[0]+large_shape[0],
                        start_lg[1]:start_lg[1]+large_shape[1],:]
        )

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            imgs = []
            for i in range(*idx.indices(len(self))):
                try:
                    imgs.append(self.load_image(i))
                except ValueError as e:
                    print(e)
            return tf.stack([img[0] for img in imgs]), tf.stack([img[1] for img in imgs])
        else:
            sm, lg = self.load_image(idx)
            return sm[tf.newaxis, :, :, :], lg[tf.newaxis, :, :, :]
    
if __name__ == '__main__':
    imdl = ScaledImageDataLoader()
    sm, lg = imdl[:10:2]
    print(sm.shape)
    print(lg.shape)
    #print(imdl.small_images)




