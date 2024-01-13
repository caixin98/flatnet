# load out and gt images, evaluate the PSNR/SSIM scores
# the data may like this:
#output_root/n12620546_7508.png
#gt_root/n12620546_7508.png

from metrics import PSNR
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import cv2