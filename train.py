import torch
import os
import numpy as np
import skimage.io

from data import GanartDataSet
from model import GanartGenerator, GanartDiscriminator

latent_size = 100
n_epochs = 100
img_shape = (256, 256, 3)
n_critique = 10
save_interval = 10
save_path = "/mnt/c/Users/davidf/workspace/ganart/out"

ds = GanartDataSet('/mnt/c/Users/davidf/workspace/ganart/circles.h5')

loader = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=True, num_workers=0)

generator = GanartGenerator(latent_size, img_shape)

discriminator = GanartDiscriminator(img_shape)

lr = 0.00005
optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=lr)
optimizer_d = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

for ni, epoch in enumerate(range(n_epochs)):
    for bi, real_imgs in enumerate(loader):
        z = np.random.uniform(0, 1, (real_imgs.shape[0], latent_size)).astype(np.float32)
        z = torch.autograd.Variable(torch.from_numpy(z))

        fake_imgs = generator(z).detach()

        real_val = discriminator(real_imgs)
        fake_val = discriminator(fake_imgs)
        
        loss_d = -torch.mean(real_val) + torch.mean(fake_val)

        loss_d.backward()
        optimizer_d.step()

        if bi % n_critique == 0:
            
            optimizer_g.zero_grad()

            gen_imgs = generator(z)
            gen_val = discriminator(gen_imgs)
            loss_g = -torch.mean(gen_val)

            loss_g.backward()
            optimizer_g.step()

        if bi % save_interval == 0:
            print(f'Epoch {ni}, Batch {bi} - saving')
            save_imgs = gen_imgs.detach().numpy()
            save_imgs = (np.clip(0,1,save_imgs)*255).astype(np.uint8)
            
            for ii in range(save_imgs.shape[0]):
                fname = os.path.join(save_path, f'image_{ni:04d}_{bi:04d}_{ii:04d}.png')
                skimage.io.imsave(fname, save_imgs[ii])
                
            

print("done")
