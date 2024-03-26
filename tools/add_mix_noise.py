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
    parser.add_argument('--eli', type=int, default=12800, help='signal to noise ratio')
    args = parser.parse_args()
    return args
def add_noise(obj_path, save_path, expected_light_intensity):
    # load images from obj_path
    if os.path.isdir(obj_path):
        for img_path in os.listdir(obj_path):
            img_path = os.path.join(obj_path, img_path)
            img = cv2.imread(img_path)
            # add noise to the images based on the snr
            # 计算信号功率
            img = img.astype(np.float32)
            signification_factor = 9800 * 64 / expected_light_intensity  
            measurement_after_poisson_noise = np.random.poisson(img * signification_factor) / signification_factor
            measurement_after_poisson_noise =  (measurement_after_poisson_noise - img) + img
            gaussian_noise_std = (2.52 / 9800 * expected_light_intensity)
            gaussian_noise = np.random.randn(*img.shape) * gaussian_noise_std
            img = measurement_after_poisson_noise + gaussian_noise
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
        img = img.astype(np.float32)
        signification_factor = 9800 * 64 / expected_light_intensity  
        measurement_after_poisson_noise = np.random.poisson(img * signification_factor) / signification_factor
        measurement_after_poisson_noise =  (measurement_after_poisson_noise - img) + img
        gaussian_noise_std = (2.52 / 9800 * expected_light_intensity)
        gaussian_noise = np.random.randn(*img.shape) * gaussian_noise_std
        img = measurement_after_poisson_noise + gaussian_noise
        #convert to 0...255
        img = img - np.min(img)
        img = img / np.max(img) * 255
        # 添加噪声到图像
        # save the images in the save_path
        os.makedirs(save_path, exist_ok=True)
        save_img_path = os.path.join(save_path, obj_path.split('/')[-1])
        cv2.imwrite(save_img_path, img)
if __name__ == "__main__":
    args = parse_args()
    obj_path = args.obj_path
    save_path = args.save_path
    eli = args.eli
    add_noise(obj_path, save_path, eli)