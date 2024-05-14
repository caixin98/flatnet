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
from models.fftlayer_diff import FFTLayer_diff
from diff_config import fft_args
from torchvision.transforms.functional import (
    to_tensor,
    resize,
)
FFT = FFTLayer_diff(fft_args)
SIZE = 270, 480

def transform(image, gray=False):
    image = np.flip(np.flipud(image), axis=2)
    image = image.copy()
    image = to_tensor(image)
    image = resize(image, SIZE)
    image = (image - 0.5) * 2
    return image

def decode_capture(FFT, obj_path, save_path):
    # load capture from obj_path with cv2
    # capture = cv2.imread(obj_path)
    # convert to tensor
    # capture = torch.tensor(capture).permute(2, 0, 1).unsqueeze(0).float()
    image = np.load(obj_path)
    print("image shape: ", image.shape)
    image = transform(image).unsqueeze(0)
    # snr_db = 20
    # #add noise to the capture based on the noise level (snr)
    # # 计算信号功率
 
    # signal_power = capture.pow(2).mean()
    # # 将SNR从dB转换为线性比例
    # snr_linear = 10 ** (snr_db / 10.0)
    # # 计算噪声功率
    # noise_power = signal_power / snr_linear
   
    # # 生成相同形状的随机噪声
    # noise = torch.randn(capture.size()) * noise_power.sqrt()
   
  
    # # 添加噪声到图像
    # capture = capture + noise

    # capture = capture + torch.randn_like(capture) * (1 / snr)
    # Change range from -1,...,1
    # capture = 2 * (capture / 255) - 1
    # pass through FFT layer
    # visualize the image 
    image_visual = image.squeeze().permute(1, 2, 0).detach().numpy()
    image_visual -= image_visual.min()
    image_visual /= image_visual.max()
    image_visual *= 255
    image_visual = image_visual.astype(np.uint8)

    cv2.imwrite("real_capture.png", image_visual)
    psf_visual = FFT.psf
    # visualize the psf
    psf_visual = psf_visual.permute(1, 2, 0).detach().numpy()
    psf_visual -= psf_visual.min()
    psf_visual /= psf_visual.max()
    psf_visual *= 255
    psf_visual = psf_visual.astype(np.uint8)
    cv2.imwrite("psf.png", psf_visual)

    print("psf_visual shape: ", psf_visual.shape)
    

    decoded = FFT(image)
    # convert to Png for visualization
    # decoded = (decoded + 1) / 2
    print("decoded min: ", decoded.min())
    print("decoded max: ", decoded.max())
    decoded = (decoded - decoded.min()) / (decoded.max() - decoded.min())
    decoded = decoded * 255
    decoded = decoded.squeeze().permute(1, 2, 0).detach().numpy().astype(np.uint8)
    # decoded = cv2.cvtColor(decoded, cv2.COLOR_RGB2BGR)
    # save image
    save_path = os.path.join(save_path, os.path.basename(obj_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_path = save_path.replace(".npy", ".png")
    print(f"Saving decoded capture to {save_path}")
    cv2.imwrite(save_path, decoded)
def decode_capture_dir(FFT, obj_dir, save_dir):
    for obj_path in Path(obj_dir).glob("*.png"):
        save_path = Path(save_dir) / obj_path.name
        save_path = str(save_path)
        obj_path = str(obj_path)
        decode_capture(FFT, obj_path, save_path)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_path", type=str, required=True)
    parser.add_argument('--save_path', default="output/decoded_rgb_capture", help='save folder path')
    args = parser.parse_args()
    obj_path = args.obj_path
    save_path = args.save_path
    # save_path = os.path.join(save_path, os.path.basename(obj_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    decode_capture(FFT, obj_path, save_path)
#python tools/decode_rgb_capture.py --obj_path data/flatnet_val/real_capture --save_path data/flatnet_val/inputs

# "/root/caixin/flatnet/data/diffusercam/diffuser_images/im2.npy"