from torch import nn
import torch
import numpy as np

class Generator(nn.Module):
    def __init__(self, input_size, image_shape, rlslope=0.2):
        super().__init__()

        self.image_shape = image_shape

        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(rlslope, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(rlslope, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(rlslope, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(rlslope, inplace=True),
            nn.Linear(1024, int(np.prod(image_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        print(img)
        img = img.view(img.shape[0], *self.image_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, image_shape, rlslope=0.2):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(image_shape)), 512),
            nn.LeakyReLU(rlslope, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(rlslope, inplace=True),
            nn.Linear(256,1)
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        return self.model(img_flat)

if __name__ == "__main__":
    import h5py
    
    fname = "/mnt/c/Users/davidf/workspace/ganart/circles.h5"

    with h5py.File(fname, "r") as f:
        data = f["data"][:10,:]


    latent_size = 100
    
    z = np.random.uniform(size=(latent_size,))
    z = np.reshape(z, (1,len(z)))
    z = torch.from_numpy(z.astype(np.float32))

    image_shape = data.shape[1:]

    generator = Generator(latent_size, image_shape)
    discriminator = Discriminator(image_shape)

    out = discriminator.forward(torch.from_numpy(data.astype(np.float32)))
    print(out.shape)
    

            
            
            
