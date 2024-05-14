# visualize the 12bit raw data
# Usage: python visual_capture.py <raw_data_folder/raw_data_file> <width> <height> <bit_depth>
# Example: python visual_capture.py 12bit.png 4096 3072 12
# save the images in the visual folder output/visual_capture + the structure as raw_data_folder
import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.ops import rggb_2_rgb

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize the 12bit raw data')
    parser.add_argument('raw_data_path', help='raw data folder / file path')
    parser.add_argument('visual_folder', default="output/visual_capture", help='visual folder path')
    # parser.add_argument('width', default=None, help='raw data width')
    # parser.add_argument('height',default=None, help='raw data height')
    parser.add_argument('bit_depth', default=12, help='raw data bit depth')
    args = parser.parse_args()
    return args
def load_raw_data(raw_data_file, bit_depth, meas_crop_size_x=1280, meas_crop_size_y=1408, meas_centre_x=808, meas_centre_y=965, psf_height=1518, psf_width=2012, pad_meas_mode="replicate"):
    # load raw png data with cv2
    bit_depth = int(bit_depth)
    print("raw_data_file: {}".format(raw_data_file))
    raw_data = cv2.imread(raw_data_file, cv2.IMREAD_UNCHANGED) / (2 ** bit_depth - 1)
    # convert to 8bit for visualization
    raw_data = raw_data.astype(np.float32)
    # reshape to the original size
    raw_h, raw_w = raw_data.shape
    img = np.zeros((raw_h // 2, raw_w // 2, 4))

    img[:, :, 0] = raw_data[0::2, 0::2]  # r
    img[:, :, 1] = raw_data[0::2, 1::2]  # gr
    img[:, :, 2] = raw_data[1::2, 0::2]  # gb
    img[:, :, 3] = raw_data[1::2, 1::2]  # b

    img = torch.tensor(img)

    # Crop
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
        img = img.squeeze(0)
        # img = img.permute(2, 0, 1)
        # img = img.squeeze(0).permute(1, 2, 0)
    # print
    # img = (img - 0.5) * 2  # Change range from -1,...,1
    # img = np.transpose(img, (2, 0, 1))
    # img = img.numpy()
    

    source_rgb = rggb_2_rgb(img)
    source_rgb = (source_rgb - source_rgb.min()) / (
    source_rgb.max() - source_rgb.min()
    )
    print('source_rgb shape: {}'.format(source_rgb.shape))
    print('source_rgb max: {}'.format(source_rgb.max()))    
    print('source_rgb dtype: {}'.format(source_rgb.dtype))
    source_rgb = source_rgb.permute(1, 2, 0)
    source_rgb = source_rgb.numpy() * 255
    source_rgb = source_rgb.astype(np.uint8)
    source_rgb = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR)
    # raw_data = raw_data.reshape((int(height), int(width)))
    return source_rgb

def save_image(raw_data, raw_data_file, visual_folder):
    if not os.path.exists(visual_folder):
        os.makedirs(visual_folder)

    visual_file = os.path.join(visual_folder, os.path.basename(raw_data_file))
    print('Saving image to {}'.format(visual_file))
    cv2.imwrite(visual_file, raw_data)
def main():
    args = parse_args()
    raw_data_path = args.raw_data_path
    visual_folder = args.visual_folder
    # width = args.width
    # height = args.height
    bit_depth = args.bit_depth
    if os.path.isdir(raw_data_path):
        # find all raw data files, including subfolders    
        raw_data_files = []
        for cls in os.listdir(raw_data_path):
            cls_path = os.path.join(raw_data_path, cls)
            if os.path.isdir(cls_path):
                for raw_data_file in os.listdir(cls_path):
                    raw_data_file_path = os.path.join(cls_path, raw_data_file)
                    if os.path.isfile(raw_data_file_path):
                        raw_data_files.append(os.path.join(cls, raw_data_file))        
        for raw_data_file in raw_data_files:
            if raw_data_file.split('/')[-1] not in os.listdir("/root/caixin/flatnet/data/flatnet/real_captures"):
                print('Processing {}'.format(raw_data_file))
                raw_data = load_raw_data(os.path.join(raw_data_path, raw_data_file), bit_depth)
                save_image(raw_data, raw_data_file, visual_folder)
    else:
        raw_data_folder = os.path.basename(os.path.dirname(os.path.dirname(raw_data_path)))
        raw_data_file = os.path.basename(raw_data_path)
        raw_data = load_raw_data(raw_data_path, bit_depth)
        save_image(raw_data, raw_data_file, raw_data_folder)
if __name__ == '__main__':
    main()
#python tools/visual_capture.py /root/caixin/data/imagenet_caps_384_12bit_Feb_19/n01440764/n01440764_457.png output/visual_capture 12