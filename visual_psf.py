from sacred import Experiment
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import logging
import cv2
from pathlib import Path
# Torch Libs
import torch
from torch.nn import functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
# Modules
from dataloader import get_dataloaders
from utils.dir_helper import dir_init
from utils.tupperware import tupperware
from models import get_model
from metrics import PSNR
from config import initialise
from skimage.metrics import structural_similarity as ssim
from utils.model_serialization import load_state_dict
import os
import cv2
# LPIPS


# Typing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.typing_alias import *

def visual_psf(matrix, save_path="visuals/psf.png"):
    print(matrix.shape)
    matrix =  (matrix - matrix.min()) / (matrix.max() - matrix.min()) * 255
    if torch.is_tensor(matrix):
        matrix = matrix.cpu().detach().numpy()
    #save
    cv2.imwrite(str(save_path), matrix)
    return matrix
  
# Experiment, add any observers by command line
ex = Experiment("val")
ex = initialise(ex)
@ex.automain
def main(_run):
    args = tupperware(_run.config)
    G, FFT, _ = get_model.model(args)
    G, FFT_init, _ = get_model.model(args)
    ckpt_dir = Path("flatnet_oss/ckpts_phase_mask_Feb_2020_size_384") / args.exp_name
    model_gen_path = ckpt_dir / "model_latest.pth"
    model_fft_path = ckpt_dir / "FFT_latest.pth"
    print("Loading model from:", model_gen_path)
    print("Loading model from:", model_fft_path)
    gen_ckpt = torch.load(model_gen_path, map_location=torch.device("cpu"))
    fft_ckpt = torch.load(model_fft_path, map_location=torch.device("cpu"))

    # G.load_state_dict(gen_ckpt["state_dict"])

    # load_state_dict(G, gen_ckpt["state_dict"])
    load_state_dict(FFT, fft_ckpt["state_dict"])
    visual_path = Path("visuals") / args.exp_name
    os.makedirs(visual_path, exist_ok=True)
    if args.multi > 1:
        for i in range(args.multi):
            save_path = visual_path / f"wiener_crop_{i}.png"
            visual_psf(FFT.wiener_crop[i], save_path=save_path)
            # calculate the difference of the wiener crop
            diff = FFT.wiener_crop[i] - FFT.wiener_crop[5]
            # print(FFT.wiener_crop[i])
            print("PSF i - 0 diff:",torch.sum(torch.abs(diff)))
            save_path = visual_path / f"wiener_crop_diff_{i}.png"
            visual_psf(diff, save_path=save_path)
    else:
        save_path = visual_path / "wiener_crop.png"
        visual_psf(FFT.wiener_crop, save_path=save_path)