# decode captures to RGB images
import argparse
import os
import cv2
import numpy as np
import torch
from pathlib import Path
import sys
import torch.nn.functional as F

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.fftlayer import FFTLayer
from config import fft_args
from utils.ops import rggb_2_rgb


def decode_raw_capture(args, obj_path, save_path):
    FFT = FFTLayer(fft_args)
    # load capture from obj_path with cv2
    capture = cv2.imread(obj_path)
    raw = cv2.imread(obj_path, -1) / 4096.0
    raw_h, raw_w = raw.shape
    img = np.zeros((raw_h // 2, raw_w // 2, 4))

    img[:, :, 0] = raw[0::2, 0::2]  # r
    img[:, :, 1] = raw[0::2, 1::2]  # gr
    img[:, :, 2] = raw[1::2, 0::2]  # gb
    img[:, :, 3] = raw[1::2, 1::2]  # b

    img = torch.tensor(img)
    # convert to tensor
    if args.meas_crop_size_x and args.meas_crop_size_y:
        crop_x = args.meas_centre_x - args.meas_crop_size_x // 2
        crop_y = args.meas_centre_y - args.meas_crop_size_y // 2

        # Replicate padding
        img = img[
            crop_x: crop_x + args.meas_crop_size_x,
            crop_y: crop_y + args.meas_crop_size_y,
            ]

        pad_x = args.psf_height - args.meas_crop_size_x
        pad_y = args.psf_width - args.meas_crop_size_y
        
        img = F.pad(
            img.permute(2, 0, 1).unsqueeze(0),
            (pad_y // 2, pad_y // 2, pad_x // 2, pad_x // 2),
            mode=args.pad_meas_mode,
        )

        img = img.squeeze(0).permute(1, 2, 0)

    img = (img - 0.5) * 2  # Change range from -1,...,1
    img = np.transpose(img, (2, 0, 1))

    # source_rgb = rggb_2_rgb(img).permute(1, 2, 0)
    # source_rgb = (source_rgb - source_rgb.min()) / (
    # source_rgb.max() - source_rgb.min()
    # )
    # pass through FFT layer
    decoded = FFT(img.unsqueeze(0).float())
    decoded = rggb_2_rgb(decoded.squeeze(0)).mul(0.5).add(0.5)
    # convert to Png for visualization
    decoded = (decoded - decoded.min()) / (decoded.max() - decoded.min())
    decoded = decoded * 255
    decoded = decoded.permute(1, 2, 0).numpy().astype(np.uint8)
    decoded = cv2.cvtColor(decoded, cv2.COLOR_RGB2BGR)
    # save image
    cv2.imwrite(save_path, decoded)
def decode_capture_dir(FFT, obj_dir, save_dir):
    for obj_path in Path(obj_dir).glob("*.png"):
        save_path = Path(save_dir) / obj_path.name
        decode_raw_capture(fft_args, obj_path, save_path)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_path", type=str, required=True)
    parser.add_argument('--save_path', default="output/decoded_raw_capture", help='save folder path')
    args = parser.parse_args()
    obj_path = args.obj_path
    save_path = args.save_path
    save_path = os.path.join(save_path, os.path.basename(obj_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    decode_raw_capture(fft_args, obj_path, save_path)

    # /root/caixin/data/imagenet_caps_384_12bit_Feb_19/n01440764/n01440764_457.png