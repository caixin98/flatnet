"""
Val Script for Phase/Amp mask
"""
# Libraries
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

# LPIPS
import lpips
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores

# Typing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.typing_alias import *

# Train helpers
from utils.ops import rggb_2_rgb, unpixel_shuffle
from utils.train_helper import load_models, AvgLoss_with_dict

# Experiment, add any observers by command line
ex = Experiment("val")
ex = initialise(ex)

# To prevent "RuntimeError: received 0 items of ancdata"
torch.multiprocessing.set_sharing_strategy("file_system")


@ex.config
def config():
    gain = 1.0
    tag = "384"


@ex.automain
def main(_run):
    args = tupperware(_run.config)
    args.batch_size = 1

    # Set device, init dirs
    device = args.device
    dir_init(args)

    # ADMM or not
    is_admm = "admm" in args.exp_name
    interm_name = "fft" if not is_admm else "admm"

    # Get data
    data = get_dataloaders(args)

    # Model
    G, FFT, _ = get_model.model(args)

    ckpt_dir = Path("ckpts_phase_mask_Feb_2020_size_384") / "ours-fft-1280-1408-learn-1280-1408-meas-1280-1408-multi"
    model_gen_path = ckpt_dir / "Epoch_90_model_latest.pth"
    model_fft_path = ckpt_dir / "Epoch_90_FFT_latest.pth"

    gen_ckpt = torch.load(model_gen_path, map_location=torch.device("cpu"))
    fft_ckpt = torch.load(model_fft_path, map_location=torch.device("cpu"))

    # G.load_state_dict(gen_ckpt["state_dict"])
    load_state_dict(G, gen_ckpt["state_dict"])
    load_state_dict(FFT, fft_ckpt["state_dict"])

    G = G.to(device)
    FFT = FFT.to(device)

    # LPIPS Criterion
    lpips_criterion = loss_fn_alex.to(device)

    # Load Models
    (G, FFT, _), _, global_step, start_epoch, loss = load_models(
        G,
        FFT,
        D=None,
        g_optimizer=None,
        fft_optimizer=None,
        d_optimizer=None,
        args=args,
        tag=args.inference_mode,
    )

    _metrics_dict = {
        "PSNR": 0.0,
        "LPIPS_01": 0.0,
        "LPIPS_11": 0.0,
        "SSIM": 0.0,
        "Time": 0.0,
    }
    avg_metrics = AvgLoss_with_dict(loss_dict=_metrics_dict, args=args)

    logging.info(
        f"Loaded experiment {args.exp_name}, dataset {args.dataset_name}, trained for {start_epoch} epochs."
    )
    if args.val_train:
        logging.info("Validating on train set.")
        data.val_loader = data.train_loader
    # Run val for an epoch
    avg_metrics.reset()
    print("len(data.val_loader)", len(data.val_loader))  
    pbar = tqdm(range(len(data.val_loader) * args.batch_size), dynamic_ncols=True)

    if args.device == "cuda:0":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    else:
        start = end = 0

    # Val and test paths
    val_path = (
        args.output_dir / f"val_{args.inference_mode}_tag_{args.tag}_gain_{args.gain}"
    )
    val_path.mkdir(exist_ok=True, parents=True)

    test_path = (
        args.output_dir / f"test_{args.inference_mode}_tag_{args.tag}_gain_{args.gain}"
    )
    test_path.mkdir(exist_ok=True, parents=True)

    with torch.no_grad():
        G.eval()
        FFT.eval()
        for i, batch in enumerate(data.val_loader):
            metrics_dict = defaultdict(float)

            source, target, filename = batch
            source, target = (source.to(device), target.to(device))

            if args.device == "cuda:0" and i:
                start.record()

            fft_output = FFT(source)

            if is_admm:
                # Upsample
                fft_output = F.interpolate(fft_output, scale_factor=4, mode="nearest")

            # Unpixelshuffle
            fft_unpixel_shuffled = unpixel_shuffle(fft_output, args.pixelshuffle_ratio)
            output_unpixel_shuffled = G(fft_unpixel_shuffled)

            output = F.pixel_shuffle(output_unpixel_shuffled, args.pixelshuffle_ratio)

            if args.device == "cuda:0" and i:
                end.record()
                torch.cuda.synchronize()
                metrics_dict["Time"] = start.elapsed_time(end)
            else:
                metrics_dict["Time"] = 0.0

            # PSNR
            metrics_dict["PSNR"] += PSNR(output, target).item()
          
            metrics_dict["LPIPS_01"] += lpips_criterion(
                output.mul(0.5).add(0.5), target.mul(0.5).add(0.5)
            ).mean().item()

            metrics_dict["LPIPS_11"] += lpips_criterion(output, target).mean().item()

            for e in range(args.batch_size):
                # Compute SSIM
                if not is_admm:
                    fft_output_vis = rggb_2_rgb(fft_output[e]).mul(0.5).add(0.5)
                else:
                    fft_output_vis = fft_output[e].mul(0.5).add(0.5)

                fft_output_vis = (fft_output_vis - fft_output_vis.min()) / (
                    fft_output_vis.max() - fft_output_vis.min()
                )

                fft_output_vis = fft_output_vis.permute(1, 2, 0).cpu().detach().numpy()

                output_numpy = (
                    output[e].mul(0.5).add(0.5).permute(1, 2, 0).cpu().detach().numpy()
                )
                target_numpy = (
                    target[e].mul(0.5).add(0.5).permute(1, 2, 0).cpu().detach().numpy()
                )
                metrics_dict["SSIM"] += ssim(
                    target_numpy, output_numpy, multichannel=True, data_range=1.0,channel_axis = -1
                )

                # Dump to output folder
                name = filename[e].replace(".JPEG", ".png")
                parent = name.split("_")[0]
                path = val_path / parent
                path.mkdir(exist_ok=True, parents=True)
                path_output = path / ("output_" + name)
                path_fft = path / (f"{interm_name}_" + name)

                cv2.imwrite(
                    str(path_output), (output_numpy[:, :, ::-1] * 255.0).astype(np.int)
                )
                cv2.imwrite(
                    str(path_fft), (fft_output_vis[:, :, ::-1] * 255.0).astype(np.int)
                )

            metrics_dict["SSIM"] = metrics_dict["SSIM"] / args.batch_size
            avg_metrics += metrics_dict

            pbar.update(args.batch_size)
            pbar.set_description(
                f"Val Epoch : {start_epoch} Step: {global_step}| PSNR: {avg_metrics.loss_dict['PSNR']:.3f} | SSIM: {avg_metrics.loss_dict['SSIM']:.3f} | LPIPS_01: {avg_metrics.loss_dict['LPIPS_01']:.3f}| LPIPS_11: {avg_metrics.loss_dict['LPIPS_11']:.3f}"
            )

        with open(val_path / "metrics.txt", "w") as f:
            L = [
                f"exp_name:{args.exp_name} trained for {start_epoch} epochs\n",
                f"Inference mode {args.inference_mode}\n",
                "Metrics \n\n",
            ]
            L = L + [f"{k}:{v}\n" for k, v in avg_metrics.loss_dict.items()]
            f.writelines(L)

        if data.test_loader:
            pbar = tqdm(
                range(len(data.test_loader) * args.batch_size), dynamic_ncols=True
            )
            for i, batch in enumerate(data.test_loader):

                source, filename = batch
                source = source.to(device)

                fft_output = FFT(source)

                if is_admm:
                    # Upsample
                    fft_output = F.interpolate(
                        fft_output, scale_factor=4, mode="nearest"
                    )

                # Unpixelshuffle
                fft_unpixel_shuffled = unpixel_shuffle(
                    fft_output, args.pixelshuffle_ratio
                )
                output_unpixel_shuffled = G(fft_unpixel_shuffled)

                output = F.pixel_shuffle(
                    output_unpixel_shuffled, args.pixelshuffle_ratio
                )

                for e in range(args.batch_size):
                    if not is_admm:
                        fft_output_vis = rggb_2_rgb(fft_output[e]).mul(0.5).add(0.5)
                    else:
                        fft_output_vis = fft_output[e].mul(0.5).add(0.5)

                    fft_output_vis = (fft_output_vis - fft_output_vis.min()) / (
                        fft_output_vis.max() - fft_output_vis.min()
                    )

                    fft_output_vis = (
                        fft_output_vis.permute(1, 2, 0).cpu().detach().numpy()
                    )

                    output_numpy = (
                        output[e]
                        .mul(0.5)
                        .add(0.5)
                        .permute(1, 2, 0)
                        .cpu()
                        .detach()
                        .numpy()
                    )
                    # Dump to output folder
                    # Phase and amplitude are nested
                    name = filename[e].replace(".JPEG", ".png")
                    parent, name = name.split("/")
                    path = test_path / parent
                    path.mkdir(exist_ok=True, parents=True)
                    path_output = path / ("output_" + name)
                    path_fft = path / (f"{interm_name}_" + name)

                    cv2.imwrite(
                        str(path_output),
                        (output_numpy[:, :, ::-1] * 255.0).astype(np.int),
                    )
                    cv2.imwrite(
                        str(path_fft),
                        (fft_output_vis[:, :, ::-1] * 255.0).astype(np.int),
                    )

                pbar.update(args.batch_size)
                pbar.set_description(f"Test Epoch : {start_epoch} Step: {global_step}")
