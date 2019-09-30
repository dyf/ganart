from model import GenartAutoencoder
import torch
import numpy as np
from torchvision.utils import save_image


def vis_random(model, N, latent_size, out_path):
    #z = np.random.normal(0, .5, (N, latent_size)).astype(np.float32)
    z = np.random.uniform(0, 1, (N, latent_size)).astype(np.float32)

    z = torch.from_numpy(z)

    imgs = model.forward_decode(z)

    save_image(imgs,
               out_path,
               nrow=3, range=[0,1])

def vis_untrained(model, img):
    pass

def main():
    latent_size = 50
    n_epochs = 500
    img_shape = (256, 256, 3)
    save_interval = 20
    lr = 0.0002
    batch_size = 40

    weights_path = 'out/model_0499.weights'
    train_data_path = './circles.h5'

    model = GenartAutoencoder(img_shape, latent_size)
    model.load_state_dict(torch.load(weights_path))    

    vis_random(model, 9, latent_size, "random.png")

if __name__ == "__main__": main()