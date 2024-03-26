# load images from obj_path
# add noise to the images based on the snr
# save the images in the save_path
import os
import cv2
import argparse
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description='Add noise to the images')
    parser.add_argument('obj_path', help='obj folder / file path')
    parser.add_argument('save_path', default="output/noisy_images", help='save folder path')
    parser.add_argument('--snr', type=int, default=20, help='signal to noise ratio')
    args = parser.parse_args()
    return args
def add_noise(obj_path, save_path, snr):
    # load images from obj_path
    if os.path.isdir(obj_path):
        for img_path in os.listdir(obj_path):
            img_path = os.path.join(obj_path, img_path)
            img = cv2.imread(img_path)
            # add noise to the images based on the snr
            # 计算信号功率
            img = img.astype(np.float32)
            signal_power = np.mean(img ** 2)
            # 将SNR从dB转换为线性比例
            snr_linear = 10 ** (snr / 10.0)
            # 计算噪声功率
            noise_power = signal_power / snr_linear
            print("noise_power: ", np.sqrt(noise_power))
            # 生成相同形状的随机噪声
            noise = np.random.randn(*img.shape) * np.sqrt(noise_power)
        
            # 添加噪声到图像
            img = img + noise
            #convert to 0...255
            img = img - np.min(img)
            img = img / np.max(img) * 255
            # save the images in the save_path
            os.makedirs(save_path, exist_ok=True)
            save_img_path = os.path.join(save_path, img_path.split('/')[-1])
            cv2.imwrite(save_img_path, img)
    else:
        img = cv2.imread(obj_path)
        # add noise to the images based on the snr
        # 计算信号功率
        signal_power = np.mean(img ** 2)
        # 将SNR从dB转换为线性比例
        snr_linear = 10 ** (snr / 10.0)
        # 计算噪声功率
        noise_power = signal_power / snr_linear
        # 生成相同形状的随机噪声
        noise = np.random.randn(*img.shape) * np.sqrt(noise_power)
        # 添加噪声到图像
        img = img + noise
        #convert to 0...255
        img = img - np.min(img)
        img = img / np.max(img) * 255
        # 添加噪声到图像
        img = img + noise
        # save the images in the save_path
        os.makedirs(save_path, exist_ok=True)
        save_img_path = os.path.join(save_path, obj_path.split('/')[-1])
        cv2.imwrite(save_img_path, img)
if __name__ == "__main__":
    args = parse_args()
    obj_path = args.obj_path
    save_path = args.save_path
    snr = args.snr
    add_noise(obj_path, save_path, snr)