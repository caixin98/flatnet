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
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.fftlayer import FFTLayer
from types import SimpleNamespace
fft_args = {
    "psf_mat": Path("data/phase_psf/sim_psf.npy"),
    "psf_height": 2012,
    "psf_width": 2012,
    "psf_centre_x": 1006,
    "psf_centre_y": 1006,
    "psf_crop_size_x": 2012,
    "psf_crop_size_y": 2012,
    "meas_height": 2012,
    "meas_width": 2012,
    "meas_centre_x": 1006,
    "meas_centre_y": 1006,
    "meas_crop_size_x": 2012,
    "meas_crop_size_y": 2012,
    "pad_meas_mode": "replicate",
    "image_height": 384,
    "image_width": 384,
    "fft_gamma": 2e4,  # Gamma for Weiner init
    "fft_requires_grad": False,
    "use_mask": False,
} 

fft_args = SimpleNamespace(**fft_args)



sensor = dict(size = np.array([4.8e-6 * 2012, 4.8e-6 * 2012]))


def parse_args():
    parser = argparse.ArgumentParser(description='Simulate the lensless capture')
    # parser.add_argument('--psf_path', default="phlatcam/psf_results/psf_sensor_sin_0.png", help='psf folder path')
    parser.add_argument('--psf_path', default="phlatcam/results/psf.png", help='psf folder path')

    parser.add_argument('--obj_path', default="phlatcam/data/obj.png", help='object folder path')
    parser.add_argument('--save_path', default="output/decode_and_sim_rgb", help='save folder path')
    parser.add_argument('--adj', help='whether to adjust the light intensity', action='store_true',default=False)

    args = parser.parse_args()
    return args
# 384 * 4.8 * 10e-6 = 0.0018432
# 2e-3 / 0.0018432 * 0.4 = 0.434

def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy, startx:startx+cropx]


def load_sim_save(simulator, obj_path, save_path, use_adjust_light_intensity=False):
    # load object
    obj = cv2.imread(obj_path)
    # obj = cv2.normalize(obj, None, 0, 255, cv2.NORM_MINMAX)
    print(obj.shape)
    # simulate
    # obj = torch.tensor(obj).permute(2, 0, 1).unsqueeze(0).float()
    img = simulator.propagate(obj)
    # normalize img
    img = img - np.min(img)
    img = img / np.max(img) * 255
    cv2.imwrite("phlatcam/data/sim_capture.png", img.astype(np.uint8)) 

    capture = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    decoded = FFT(capture)
    decoded = (decoded - decoded.min()) / (decoded.max() - decoded.min())
    decoded = decoded * 255
    decoded = decoded.squeeze().permute(1, 2, 0).numpy().astype(np.uint8)    


    save_path = os.path.join(save_path, os.path.basename(obj_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"save_path: {save_path}")
    cv2.imwrite(save_path, decoded)
   

if __name__ == "__main__":
    args = parse_args()
    psf_path = args.psf_path
    obj_path = args.obj_path
    save_path = args.save_path
    adj = args.adj

    # load psf
    psf = cv2.imread(psf_path)
    psf = cv2.cvtColor(psf, cv2.COLOR_BGR2GRAY)
    # add last dimension
    psf = cv2.resize(psf, (2012, 2012))

    np.save("data/phase_psf/sim_psf.npy", psf)
    psf = psf[..., None]
    print(psf.shape)
    # transfer the psf 
    simulator = FarFieldSimulator(object_height = 0.4, scene2mask = 0.434, mask2sensor = 2e-3, sensor = sensor, psf = psf, is_torch=False, quantize=False, return_float=True)
    FFT = FFTLayer(fft_args)

    if os.path.isdir(obj_path):
        obj_path_list = os.listdir(obj_path)
        obj_path_list = [os.path.join(obj_path, obj_path_i) for obj_path_i in obj_path_list]
        for obj_path_i in obj_path_list:
            load_sim_save(simulator, obj_path_i, save_path, use_adjust_light_intensity=adj)
    else:
        load_sim_save(simulator, obj_path, save_path, use_adjust_light_intensity=adj)

# python tools/decode_and_sim_rgb.py --psf_path data/phase_psf/psf.npy --obj_path /root/caixin/StableSR/data/flatnet_val/gts --save_path /root/caixin/StableSR/data/flatnet_val/sim_captures
# python tools/decode_and_sim_rgb.py --obj_path  /root/caixin/StableSR/data/flatnet/gts/n01440764_457.png

