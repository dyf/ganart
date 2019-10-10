import torch
import os
import numpy as np
import skimage.io

from data import GenartDataSet
from model import GenartGenerator, GenartDiscriminator
from torch.autograd import Variable

from torchvision.utils import save_image

Tensor = torch.FloatTensor

latent_size = 10
n_epochs = 100
img_shape = (256, 256, 3)
save_interval = 5
lr = 0.0002

save_path = "./out"

ds = GenartDataSet('./circles.h5')

loader = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=True, num_workers=0)

generator = GenartGenerator(latent_size, img_shape)

discriminator = GenartDiscriminator(img_shape)

optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5,0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5,0.999))
adversarial_loss = torch.nn.BCELoss()

for ni, epoch in enumerate(range(n_epochs)):
    for bi, (imgs,_) in enumerate(loader):
        # ground truths
        valid = Variable(Tensor(imgs.size(0),1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0),1).fill_(0.0), requires_grad=False)

        real_imgs = Variable(imgs.type(Tensor))

        # train generator
        # ---------------
        optimizer_g.zero_grad()

        # noise input
        z = Variable(Tensor(np.random.normal(0, 1, (real_imgs.shape[0], latent_size))))

        fake_imgs = generator(z)

        gen_loss = adversarial_loss(discriminator(fake_imgs), valid)

        gen_loss.backward()

        optimizer_g.step()

        # train discriminator
        # -------------------

        real_vals = discriminator(real_imgs)
        real_loss = adversarial_loss(real_vals, valid)

        fake_vals = discriminator(fake_imgs.detach())
        fake_loss = adversarial_loss(fake_vals, fake)

        d_loss = (real_loss + fake_loss) * 0.5

        d_loss.backward()

        optimizer_d.step()
        
        if bi % save_interval == 0:
            print(f'Epoch {ni}, Batch {bi} - saving')
            save_image(fake_imgs.data[:9],
                       os.path.join(save_path, f'images_{ni:04d}_{bi:04d}.png'),
                       nrow=3, range=[0,1])
            save_image(real_imgs.data[:9],
                       os.path.join(save_path, f'real_images_{ni:04d}_{bi:04d}.png'),
                       nrow=3, range=[0,1])
                
            

print("done")
