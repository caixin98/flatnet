# compare simulation and real rgb images
# first, we need to load the images
# then, we need to transform them to grayscale and [0, 255] range
# finally, we can compare the images by absolute difference and save the result
import cv2
import numpy as np
import os
import argparse

def load_images(sim_path, real_path):
    sim_img = cv2.imread(sim_path)
    real_img = cv2.imread(real_path)
    return sim_img, real_img
def transform_images(sim_img, real_img):
    sim_img = cv2.cvtColor(sim_img, cv2.COLOR_BGR2GRAY)
    real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
    sim_img = cv2.normalize(sim_img, None, 0, 255, cv2.NORM_MINMAX)
    real_img = cv2.normalize(real_img, None, 0, 255, cv2.NORM_MINMAX)
    return sim_img, real_img
def compare_images(sim_img, real_img, output_path):
    # diff_img = cv2.absdiff(sim_img, real_img)
    sim_img = sim_img.astype(np.float32)
    real_img = real_img.astype(np.float32)
    diff_img = sim_img - real_img
    # diff_img = cv2.subtract(real_img, sim_img)
    # diff_img = cv2.normalize(diff_img, None, 0, 255, cv2.NORM_MINMAX)
    print(f'Max difference: {np.max(diff_img)}')
    print(f'Min difference: {np.min(diff_img)}')
    print(f'Mean difference: {np.mean(diff_img)}')
    diff_img = diff_img - np.min(diff_img)
    diff_img = cv2.normalize(diff_img, None, 0, 255, cv2.NORM_MINMAX)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, diff_img)
    return diff_img
def argument_parser():
    parser = argparse.ArgumentParser(description='Compare simulation and real images')
    parser.add_argument('--sim_path', default='/root/caixin/flatnet/output/sim_capture/n01440764_457.png', help='Path to the simulation image')
    parser.add_argument('--real_path', default='output/visual_capture/imagenet_caps_384_12bit_Feb_19/n01440764_457.png', help='Path to the real image')
    parser.add_argument('--output_path', default='output/compare_sim_real/n01440764_457.png', help='Path to the output image')
    return parser
def main():
    parser = argument_parser()
    args = parser.parse_args()
    sim_img, real_img = load_images(args.sim_path, args.real_path)
    sim_img, real_img = transform_images(sim_img, real_img)
    diff_img = compare_images(sim_img, real_img, args.output_path)
    print(f'Difference image saved to {args.output_path}')
if __name__ == '__main__':
    main()