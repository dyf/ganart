import imageio
import os

png_dir = './chinese_output/'
images = []
file_names = os.listdir(png_dir)

def sortkey(x):
    o = x[:-4].split('_')[-1]
    return int(o)

file_names = sorted([f for f in file_names if f.endswith('.png')], key=sortkey)

for file_name in file_names:    
   file_path = os.path.join(png_dir, file_name)
   images.append(imageio.imread(file_path))
imageio.mimsave('testa.gif', images)