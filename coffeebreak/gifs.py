import os
import imageio

images = []
file_names = os.listdir('./temp/')

for filename in file_names:
    images.append(imageio.imread('./temp/' + filename))

imageio.mimsave('./neuron_5.gif', images, duration=0.5)
