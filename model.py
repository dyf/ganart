from torch import nn
import torch
import numpy as np

class GanartGenerator(nn.Module):
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

        self.conv_init_size = image_shape[0] // 4
        self.conv_input_layer = nn.Sequential(
            nn.Linear(input_size, 128 * self.conv_init_size ** 2)
        )
        self.conv_layers = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(rlslope, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(rlslope, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )
            
            

    def forward(self, z):
        #img = self.model(z)
        #img = img.view(img.shape[0], *self.image_shape)

        out = self.conv_input_layer(z)
        out = out.view(out.shape[0], 128, self.conv_init_size, self.conv_init_size)
        img = self.conv_layers(out)
        return img


class GanartDiscriminator(nn.Module):
    def __init__(self, image_shape, rlslope=0.2):
        super().__init__()

        
        self.model_linear = nn.Sequential(            
            nn.Linear(int(np.prod(image_shape)), 512),
            nn.LeakyReLU(rlslope, inplace=True),
            nn.Dropout(.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(rlslope, inplace=True),
            nn.Dropout(.2),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

        self.model_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.LeakyReLU(rlslope, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.LeakyReLU(rlslope, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(rlslope, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(rlslope, inplace=True),
            nn.Dropout2d(0.25),            
        )

        self.conv_adv_layer = nn.Sequential(
            nn.Linear(32768, 1),
            nn.Sigmoid()
        )


    def forward(self, img):
        #img_flat = img.view(img.shape[0], -1)
        #return self.model(img_flat)

        out = self.model_conv(img)
        out = out.view(out.shape[0], -1)
        val = self.conv_adv_layer(out)
        return val

        

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
    

            
            
            
