# visualize the diffusercam image samples
from PIL import Image

from datasets.diffusercam import LenslessLearningCollection
import os
import numpy as np
image_dir = 'data/diffusercam'
class Args:
    image_dir = image_dir
# 创建 Args 的一个实例
args = Args()
dataset = LenslessLearningCollection(args)
train_dataset = dataset.train_dataset
os.makedirs("temp", exist_ok=True)
for i in range(3):
    x, y = train_dataset[i]
    #from tensor to numpy
    x = x.numpy().transpose(1, 2, 0)
    y = y.numpy().transpose(1, 2, 0)
    #normalize
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())

    x_uint8 = (x * 255).astype(np.uint8)
    y_uint8 = (y * 255).astype(np.uint8)
    print(x.shape, y.shape)
    Image.fromarray(x_uint8).save(f"temp/diffusercam_{i}_x.png")
    Image.fromarray(y_uint8).save(f"temp/diffusercam_{i}_y.png")