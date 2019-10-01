from model import GenartAutoencoder
from data import GenartDataSet
import torch
import numpy as np
from torchvision.utils import save_image
import random
import skimage.io as skio

def vis_random(model_idxs, N, img_shape, latent_size, out_path):
    z = np.random.uniform(0, 1, (N, latent_size)).astype(np.float32)
    z = torch.from_numpy(z) 

    for model_idx in model_idxs:
        try:
            model = load_autoencoder(model_idx, img_shape, latent_size)            
        except FileNotFoundError:
            continue
        #z = np.random.normal(0, .5, (N, latent_size)).astype(np.float32)    

        print(model_idx)

        imgs = model.forward_decode(z)

        save_image(imgs,
                   out_path % model_idx,
                   nrow=3, range=[0,1])

def vis_trained(model_idxs, img_idxs, img_shape, latent_size, train_data_path, out_path):
    ds = GenartDataSet(train_data_path)
    
    imgs,_ = ds[img_idxs]
    imgs = torch.from_numpy(imgs.astype(np.float32))
    print(imgs.shape)

    for model_idx in model_idxs:
        try:
            model = load_autoencoder(model_idx, img_shape, latent_size)            
        except FileNotFoundError:
            continue

        out_imgs = model.forward(imgs)

        save_image(out_imgs,
                   out_path % model_idx,
                   nrow=3, range=[0,1])


def vis_untrained(model_idxs, img, img_shape, latent_size, out_path):
    img = torch.from_numpy(np.array([img]))

    for model_idx in model_idxs:
        try:
            model = load_autoencoder(model_idx, img_shape, latent_size)            
        except FileNotFoundError:
            continue
        print(img.shape)
        out_imgs = model.forward(img)

        save_image(out_imgs,
                   out_path % model_idx,
                   range=[0,1])

        

def load_autoencoder(i, img_shape, latent_size):
    weights_path = f'out/model_{i:04d}.weights'
    
    model = GenartAutoencoder(img_shape, latent_size)
    model.load_state_dict(torch.load(weights_path))    

    return model

def main():
    latent_size = 50
    n_epochs = 500
    img_shape = (256, 256, 3)
    save_interval = 20
    lr = 0.0002
    batch_size = 40

    #vis_random(range(500), 9, img_shape, latent_size, "vis/random_%04d.png")
    #vis_trained(range(500), sorted(random.choices(range(10000),k=9)), img_shape, latent_size, "./circles.h5", "vis/trained_%04d.png")
    img = skio.imread('cat.jpg').transpose((2,0,1)).astype(np.float32) / 255.0
    vis_untrained(range(500), img, img_shape, latent_size, "vis/untrained_%04d.png")

if __name__ == "__main__": main()