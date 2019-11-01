import tensorflow as tf
from pathlib import Path
import re

class ScaledImageDataLoader:
    def __init__(self, basedir='./images'):
        self.basedir = Path(basedir)

        self.small_images = list(self.basedir.glob("small/img-small*"))
        self.large_images = list(self.basedir.glob("large/img-large*"))
        
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
        small_image = tf.image.decode_jpeg(f)
        f = tf.io.read_file(str(self.basedir / 'large' / f'img-large-{idx}.jpg'))
        large_image = tf.image.decode_jpeg(f)

        return self.square_image(small_image), self.square_image(large_image)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            imgs = [ self.load_image(i) for i in range(*idx.indices(len(self))) ]
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



