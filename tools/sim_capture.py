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
    img = img.numpy()
    return img
    
if __name__ == "__main__":
    args = parse_args()
    psf_path = args.psf_path
    obj_path = args.obj_path
    save_path = args.save_path

    # load psf
    psf = np.load(psf_path)
    # add last dimension
    psf = psf[..., None]
    print(psf.shape)
    # transfer the psf 
    simulator = FarFieldSimulator(object_height = 0.4, scene2mask = 0.434, mask2sensor = 2e-3, sensor = sensor, psf = psf, is_torch=False)
    # load object
    obj = cv2.imread(obj_path)
    # simulate
    img, object_plane = simulator.propagate(obj,return_object_plane=True)
    img = crop_and_padding(img)
    # save
    # save_path = os.path.join(save_path, os.path.basename(psf_path).split(".")[0], os.path.basename(obj_path))
    save_path = os.path.join(save_path, os.path.basename(obj_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # img = img.permute(1, 2, 0).numpy().astype(np.uint8)
    cv2.imwrite(save_path, img)
    # save object plane
    # save_path = save_path.replace(".png", "_object_plane.png")
    # cv2.imwrite(save_path, object_plane)
    # print("save image to {}".format(save_path))

