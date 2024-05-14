#resize dataset to 512x512
import os
import cv2
import numpy as np
from tqdm import tqdm
# dataset_path = '/root/caixin/StableSR/data/flatnet/decoded_sim_captures'
# resize_path = "/root/caixin/StableSR/data/flatnet/decoded_sim_captures"
dataset_path = '/root/caixin/StableSR/data/flatnet_val/inputs'
resize_path = "/root/caixin/StableSR/data/flatnet_val/inputs_512"
resize_size = (512, 512)
os.makedirs(resize_path, exist_ok=True)
for img_name in tqdm(os.listdir(dataset_path)):
    if not img_name.endswith('.png'):
        continue
    # if os.path.exists(os.path.join(resize_path, img_name)):
    #     continue
    
    img_path = os.path.join(dataset_path, img_name)
    img = cv2.imread(img_path)
    img = cv2.resize(img, resize_size)
    cv2.imwrite(os.path.join(resize_path, img_name), img)
    print(f"Resized {img_path} to {resize_size}")