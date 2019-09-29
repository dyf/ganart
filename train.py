import torch
import os
import numpy as np
import skimage.io

from data import GanartDataSet
from model import GanartAutoencoder
from torch.autograd import Variable

from torchvision.utils import save_image

Tensor = torch.FloatTensor

latent_size = 50
n_epochs = 100
img_shape = (256, 256, 3)
save_interval = 5
lr = 0.0002

save_path = "/mnt/c/Users/davidf/workspace/ganart/out"

ds = GanartDataSet('/mnt/c/Users/davidf/workspace/ganart/circles.h5')

loader = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=True, num_workers=0)

model = GanartAutoencoder(img_shape, latent_size)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5,0.999))
loss = torch.nn.MSELoss()

for ni, epoch in enumerate(range(n_epochs)):
    for bi, (imgs,_) in enumerate(loader):
        optimizer.zero_grad()

        # noise input
        out_imgs = model(imgs)
        batch_loss = loss(out_imgs, imgs)
        batch_loss.backward()
        optimizer.step()

        if bi % save_interval == 0:
            print(f'Epoch {ni}, Batch {bi} - saving')
            save_image(out_imgs.data[:9],
                       os.path.join(save_path, f'images_{ni:04d}_{bi:04d}.png'),
                       nrow=3, range=[0,1])
print("done")
