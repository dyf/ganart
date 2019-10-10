import imageio

idxs = range(0, 410, 10)
img_paths = [ f'vis_v3/trained_{idx:04d}.png' for idx in idxs ]

imgs =  [ imageio.imread(img_path) for img_path in img_paths ]

imageio.mimsave('vis_v3/trained.gif', imgs, fps=5)