from waveprop.simulation import FarFieldSimulator
# output the lensless capture with a given psf and a given object, and save the image
# Usage: python sim_capture.py --psf_path psf.png --obj_path obj.png 
# save the images in the visual folder output/sim_capture/psf_folder/obj_folder
import argparse
import os 
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import sys
from pathlib import Path
from waveprop.devices import  SensorParam

sensor = dict(size = np.array([4.8e-6 * 1518, 4.8e-6 * 2012]))

# class FarFieldSimulator(object):

    # """
    # Simulate far-field propagation with the following steps:
    # 1. Resize digital image for desired object height and to PSF resolution.
    # 2. Convolve with PSF
    # 3. (Optionally) Resize to lower sensor resolution.
    # 4. (Optionally) Add shot noise
    # 5. Quantize


    # Images and PSFs should be one of following shape
    # - For numpy arrays: (H, W) for grayscale and (H, W, 3) for RGB.
    # - For PyTorch tensors: (..., H, W)

    # """

    # def __init__(
    #     self,
    #     object_height,
    #     scene2mask,
    #     mask2sensor,
    #     sensor,
    #     psf=None,
    #     output_dim=None,
    #     snr_db=None,
    #     max_val=255,
    #     device_conv="cpu",
    #     random_shift=False,
    #     is_torch=False,
    #     quantize=True,
    #     return_float=True,
    #     **kwargs
    # ):

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

def parse_args():
    parser = argparse.ArgumentParser(description='Simulate the lensless capture')
    parser.add_argument('--psf_path', default="data/phase_psf/psf.npy", help='psf folder path')
    parser.add_argument('--obj_path', default=None, help='object folder path')
    parser.add_argument('--save_path', default="output/sim_capture", help='save folder path')
    parser.add_argument('--adj', help='whether to adjust the light intensity', action='store_true',default=False)

    args = parser.parse_args()
    return args
# 384 * 4.8 * 10e-6 = 0.0018432
# 2e-3 / 0.0018432 * 0.4 = 0.434

def crop_and_padding(img, meas_crop_size_x=1280, meas_crop_size_y=1408, meas_centre_x=808, meas_centre_y=965, psf_height=1518, psf_width=2012, pad_meas_mode="replicate"):
    # crop
    img = torch.tensor(img)
    if meas_crop_size_x and meas_crop_size_y:
        crop_x = meas_centre_x - meas_crop_size_x // 2
        crop_y = meas_centre_y - meas_crop_size_y // 2

        # Replicate padding
        img = img[
            crop_x: crop_x + meas_crop_size_x,
            crop_y: crop_y + meas_crop_size_y,
            ]

        pad_x = psf_height - meas_crop_size_x
        pad_y = psf_width - meas_crop_size_y
        
        img = F.pad(
            img.permute(2, 0, 1).unsqueeze(0),
            (pad_y // 2, pad_y // 2, pad_x // 2, pad_x // 2),
            mode=pad_meas_mode,
        )

        img = img.squeeze(0).permute(1, 2, 0)
        # resize for half
    img = img.numpy()
    # img = cv2.resize(img.numpy(), (meas_crop_size_y // 2, meas_crop_size_x // 2))
    
    # img = img[..., None]
    print("img shape: ", img.shape)
    return img
# adjust the light intensity of img (add weight 0.6-1 on the input img), the center of the img is the brightest
def adjust_light_intensity(image, min_weight=0.2):
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the center of the image
    center_x, center_y = width / 2, height / 2

    # Create the vignette mask 
    # Generate a grid of the same size as the image
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    X = (X - center_x) / center_x
    Y = (Y - center_y) / center_y

    # Calculate the distance from the center to each pixel
    distance = np.sqrt(X**2 + Y**2)

    # Generate the weight mask by using a circular pattern that attenuates towards the edges
    # max_distance = np.sqrt(center_x**2 + center_y**2)
    max_distance = np.sqrt(1**2 + 1**2)
    weights = 1 - distance / max_distance * (1 - min_weight)
    
    # weights = np.clip(1 - distance / max_distance, 0.6, 1)

    # Apply a weight factor that increases towards the center (the vignette effect)
    image = image.astype(np.float32)
    weights = weights.astype(np.float32)
    for i in range(3):  # Apply for each channel of the image
        image[:, :, i] *= weights
    image = np.clip(image, 0, 255).astype(np.uint8)
    # save_path = os.path.join("output/rgb_adj","adj.png")
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # cv2.imwrite(save_path, image)
    # # visualize the weight
    # weights = weights * 255
    # weights = weights.astype(np.uint8)
    # save_path = os.path.join("output/rgb_adj","weights.png")
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # cv2.imwrite(save_path, weights)
    return image
    


def load_sim_save(simulator, obj_path, save_path, use_adjust_light_intensity=False):
    # load object
    obj = cv2.imread(obj_path)
    obj = cv2.normalize(obj, None, 0, 255, cv2.NORM_MINMAX)
    if use_adjust_light_intensity:
        print("adjust light intensity")
        obj = adjust_light_intensity(obj)
    # save_adj_path = os.path.join("output/rgb_adj","adj%s.png"%use_adjust_light_intensity)
    # os.makedirs(os.path.dirname(save_adj_path), exist_ok=True)
    # cv2.imwrite(save_adj_path, obj)
        # print("the difference between the original and the adjusted image is: ", np.sum(obj_ - obj))
    # simulate
    img, object_plane = simulator.propagate(obj,return_object_plane=True)
    #normalize img
  
    img = img - np.min(img)
    img = img / np.max(img) * 255

    # img_, object_plane = simulator.propagate(obj_,return_object_plane=True)
    # #normalize img
    # img_ = img_ - np.min(img_)
    # img_ = img_ / np.max(img_) * 255
    # print("the difference between the original and the adjusted image is: ", np.sum(img_ - img))

    save_path = os.path.join(save_path, os.path.basename(obj_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"save_path: {save_path}")
    cv2.imwrite(save_path, img)
   

if __name__ == "__main__":
    args = parse_args()
    psf_path = args.psf_path
    obj_path = args.obj_path
    save_path = args.save_path
    adj = args.adj

    # load psf
    psf = np.load(psf_path)
    # add last dimension
    psf = psf[..., None]
    print(psf.shape)
    psf = crop_and_padding(psf)
    # transfer the psf 
    simulator = FarFieldSimulator(object_height = 0.4, scene2mask = 0.434, mask2sensor = 2e-3, sensor = sensor, psf = psf, is_torch=False)
    print("adj: ", adj)
    if os.path.isdir(obj_path):
        obj_path_list = os.listdir(obj_path)
        obj_path_list = [os.path.join(obj_path, obj_path_i) for obj_path_i in obj_path_list]
        for obj_path_i in obj_path_list:
            load_sim_save(simulator, obj_path_i, save_path, use_adjust_light_intensity=adj)
    else:
        load_sim_save(simulator, obj_path, save_path, use_adjust_light_intensity=adj)

# python tools/sim_capture.py --psf_path data/phase_psf/psf.npy --obj_path /root/caixin/StableSR/data/flatnet_val/gts --save_path /root/caixin/StableSR/data/flatnet_val/sim_captures
# python tools/sim_capture.py --obj_path  /root/caixin/StableSR/data/flatnet/gts/n01440764_457.png

