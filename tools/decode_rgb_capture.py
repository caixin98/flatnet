# decode captures to RGB images
import argparse
import os
import cv2
import numpy as np
import torch
from pathlib import Path
import sys

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.fftlayer import FFTLayer
from config import fft_args
FFT = FFTLayer(fft_args)

def decode_capture(FFT, obj_path, save_path):
    # load capture from obj_path with cv2
    capture = cv2.imread(obj_path)
    # convert to tensor
    capture = torch.tensor(capture).permute(2, 0, 1).unsqueeze(0).float()
    # Change range from -1,...,1
    capture = 2 * (capture / 255) - 1
    # pass through FFT layer
    decoded = FFT(capture)
    # convert to Png for visualization
    decoded = (decoded + 1) / 2
    decoded = (decoded - decoded.min()) / (decoded.max() - decoded.min())
    decoded = decoded * 255
    decoded = decoded.squeeze().permute(1, 2, 0).numpy().astype(np.uint8)
    # decoded = cv2.cvtColor(decoded, cv2.COLOR_RGB2BGR)
    # save image
    cv2.imwrite(save_path, decoded)
def decode_capture_dir(FFT, obj_dir, save_dir):
    for obj_path in Path(obj_dir).glob("*.png"):
        save_path = Path(save_dir) / obj_path.name
        decode_capture(FFT, obj_path, save_path)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_path", type=str, required=True)
    parser.add_argument('--save_path', default="output/decoded_rgb_capture", help='save folder path')
    args = parser.parse_args()
    obj_path = args.obj_path
    save_path = args.save_path
    save_path = os.path.join(save_path, os.path.basename(obj_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    decode_capture(FFT, obj_path, save_path)