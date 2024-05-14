# visualize the diffusercam image samples
# from .npy to .png

import numpy as np
original_data_path = "/root/caixin/flatnet/data/diffusercam/diffuser_images"
saved_data_path = "/root/caixin/flatnet/data/diffusercam/diffuser_images_png"
import os
import cv2


if not os.path.exists(saved_data_path):
    os.makedirs(saved_data_path)
for file in os.listdir(original_data_path):
    if file.endswith(".npy"):
        data = np.load(os.path.join(original_data_path, file))
        data = np.flip(np.flipud(data), axis=2)
        # data = data.squeeze().permute(1, 2, 0).numpy()
        data = (data - data.min()) / (data.max() - data.min())
        data = data * 255
        data = data.astype(np.uint8)
        #from BGR to RGB
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(saved_data_path, file.replace(".npy", ".png")), data)
print("Done")