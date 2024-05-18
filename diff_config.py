"""
Convention

ours/naive-fft-(fft_h-fft_w)-learn-(learn_h-learn_w)-meas-(meas_h-meas_w)-kwargs

* diffusercam: 1080 x 1920 (post demosiacking)
"""
from pathlib import Path
import torch
from types import SimpleNamespace

def base_config():
    exp_name = "ours-fft-1280-1408-diff"
    is_admm = "admm" in exp_name
    is_naive = "naive" in exp_name
    multi = 1
    use_spatial_weight = False
    weight_update = True
    dataset = "diffusercam"
    # ---------------------------------------------------------------------------- #
    # Directories
    # ---------------------------------------------------------------------------- #


    image_dir = Path("data/diffusercam")
    output_dir = Path("output_diffusercam") / exp_name
    ckpt_dir = Path("ckpts_diffusercam") / exp_name
    run_dir = Path("runs_diffusercam") / exp_name  # Tensorboard

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    dataset_name = "diffusercam"

    shuffle = True
    train_gaussian_noise = 0.0

    # ---------------------------------------------------------------------------- #
    # Image, Meas, PSF Dimensions
    # ---------------------------------------------------------------------------- #
    # PSF
    psf_mat = Path("data/diffusercam/psf.tiff")

    meas_height = 270  # original height of meas
    meas_width = 480  # original width of meas
    decode_sim = False

    meas_centre_x = 135
    meas_centre_y = 240

    psf_height = 270  # fft layer height
    psf_width = 480  # fft layer width
    psf_crop_size_x = 270
    psf_crop_size_y = 480
    psf_centre_x = meas_centre_x
    psf_centre_y = meas_centre_y
    meas_crop_size_x = 270
    meas_crop_size_y = 480
    fft_gamma = 5000
    use_mask = False
    # pad meas
    pad_meas_mode = "replicate" if not is_admm else "constant"  # If none, no padding
    preprocess_with_unet = False

    image_height = 270
    image_width = 480
    model = "UNet270480"
    batch_size = 18
    num_threads = batch_size >> 1  # parallel workers

    # ---------------------------------------------------------------------------- #
    # Train Configs
    # ---------------------------------------------------------------------------- #
    # Schedules
    num_epochs = 100
    fft_epochs = num_epochs if is_naive else 0

    learning_rate = 1e-4
    fft_learning_rate = 3e-5 if not is_admm else 3e-5

    # Betas for AdamW. We follow https://arxiv.org/pdf/1704.00028
    beta_1 = 0.9  # momentum
    beta_2 = 0.999

    lr_scheduler = "cosine"  # or step

    # Cosine annealing
    T_0 = 1
    T_mult = 2
    step_size = 2  # For step lr

    # saving models
    save_filename_G = "model.pth"
    save_filename_FFT = "FFT.pth" if not is_admm else "ADMM.pth"
    save_filename_D = "D.pth"

    save_filename_latest_G = "model_latest.pth"
    save_filename_latest_FFT = "FFT_latest.pth" if not is_admm else "ADMM_latest.pth"
    save_filename_latest_D = "D_latest.pth"

    log_interval = 100  # the number of iterations (default: 10) to print at
    save_ckpt_interval = log_interval * 10
    save_copy_every_epochs = 10
    # ---------------------------------------------------------------------------- #
    # Model
    # ---------------------------------------------------------------------------- #
    # See models/get_model.py for registry
    # model = "unet-128-pixelshuffle-invert"
    pixelshuffle_ratio = 2
    grad_lambda = 0.0
    # admm model args
    admm_iterations = 5
    normalise_admm_psf = False

    G_finetune_layers = []  # None implies all

    num_groups = 8  # Group norm

    # ---------------------------------------------------------------------------- #
    # Loss
    # ---------------------------------------------------------------------------- #
    lambda_adversarial = 0.6
    lambda_contextual = 0.0
    lambda_perception = 1.2  # 0.006
    lambda_image = 1  # mse
    lambda_l1 = 0 # l1

    resume = False
    finetune = False  # Wont load loss or epochs
    concat_input = False
    zero_conv = False
    # ---------------------------------------------------------------------------- #
    # Inference Args
    # ---------------------------------------------------------------------------- #
    inference_mode = "latest"
    assert inference_mode in ["latest", "best"]

    # ---------------------------------------------------------------------------- #
    # Distribution Args
    # ---------------------------------------------------------------------------- #
    # choose cpu or cuda:0 device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    distdataparallel = False
    val_train = False
    static_val_image = ""
def ours_diffusercam():
    exp_name = "fft-diffusercam"
    # learning_rate = 3e-4
    # fft_learning_rate = 4e-10
    batch_size = 10
    num_threads = 5
    lambda_adversarial = 0.0
    # val_train = True
def le_admm_diffusercam():
    exp_name = "le-admm-fft-diffusercam"
    lambda_adversarial = 0.0
    psf_mat = Path("data/diffusercam/psf_flip.npy")
    batch_size = 5
    num_threads = 5
    num_epochs = 30


def ours_diffusercam_concat_input():
    exp_name = "fft-diffusercam_concat_input"
    # learning_rate = 3e-4
    # fft_learning_rate = 4e-10
    batch_size = 10
    num_threads = 5
    lambda_adversarial = 0.0
    concat_input = True
    # val_train = True

def ours_diffusercam_decoded_sim():
    exp_name = "fft-mulnew9-diffusercam_unet_decoded_sim_val_train"
    val_train = True
    batch_size = 5
    num_threads = 5
    # val_train = True
    lambda_adversarial = 0.0
    multi = 10
    use_spatial_weight = True
    # resume = True
    # finetune = True  # Wont load loss or epochs
    lambda_perception = 0.05
    preprocess_with_unet = True
    decode_sim = True
    ckpt_dir = Path("ckpts_diffusercam") / exp_name.replace("_val_train", "")




def ours_diffusercam_multi():
    exp_name = "fft-multi9-diffusercam"
    # train_target_list =  "data/text_files/decoded_sim_captures_train.txt"
    # val_target_list = "data/text_files/decoded_sim_captures_val.txt"
    # learning_rate = 3e-4
    # fft_learning_rate = 4e-10
    batch_size = 5
    num_threads = 5
    # val_train = True
    lambda_adversarial = 0.6
    multi = 10


def ours_diffusercam_mulnew():
    exp_name = "fft-mulnew9-diffusercam"
    # val_train = True
    batch_size = 5
    num_threads = 5
    # val_train = True
    lambda_adversarial = 0.0
    multi = 10
    use_spatial_weight = True
    resume = True
    finetune = True  # Wont load loss or epochs
    lambda_perception = 0.05
    # ckpt_dir = Path("ckpts_diffusercam") / exp_name.replace("_val_train", "")

def ours_diffusercam_mulnew_unet():
    exp_name = "fft-mulnew9-diffusercam_unet"
    # val_train = True
    batch_size = 5
    num_threads = 5
    # val_train = True
    lambda_adversarial = 0.0
    multi = 10
    use_spatial_weight = True
    # resume = True
    # finetune = True  # Wont load loss or epochs
    lambda_perception = 0.05
    preprocess_with_unet = True


def ours_diffusercam_mulnew_unet_padding():
    exp_name = "fft-mulnew9-diffusercam_unet_padding"
    # val_train = True
    batch_size = 5
    num_threads = 5
    # val_train = True
    lambda_adversarial = 0.0
    multi = 10
    use_spatial_weight = True
    # resume = True
    # finetune = True  # Wont load loss or epochs
    lambda_perception = 0.05
    preprocess_with_unet = True
    psf_height = 270 * 2
    psf_width = 480 * 2
    num_epochs = 40

    

def ours_diffusercam_mulnew_unet_padding_decode_sim():
    exp_name = "fft-mulnew9-diffusercam_unet_padding_decode_sim_val_train"
    val_train = True
    batch_size = 5
    num_threads = 5
    lambda_adversarial = 0.0
    multi = 10
    use_spatial_weight = True
    # resume = True
    # finetune = True  # Wont load loss or epochs
    lambda_perception = 0.05
    preprocess_with_unet = True
    psf_height = 270 * 2
    psf_width = 480 * 2
    decode_sim = True
    num_epochs = 40
    ckpt_dir = Path("ckpts_diffusercam") / exp_name.replace("_val_train", "")

def ours_diffusercam_mulnew_unet_zero_conv():
    exp_name = "fft-mulnew9-diffusercam_unet_zero_conv"
    # val_train = True
    batch_size = 5
    num_threads = 5
    # val_train = True
    lambda_adversarial = 0.0
    multi = 10
    use_spatial_weight = True
    weight_update = False
    # resume = True
    # finetune = True  # Wont load loss or epochs
    lambda_perception = 0.05
    preprocess_with_unet = True
    zero_conv = True

def ours_diffusercam_mulnew_concat_input():
    exp_name = "fft-mulnew9-diffusercam-concat_input"
  
    batch_size = 5
    num_threads = 5
    # val_train = True
    lambda_adversarial = 0.0
    multi = 10
    use_spatial_weight = True
    concat_input = True
 

def ours_diffusercam_decoded_sim_mulnew9():
    exp_name = "fft-mulnew9-diffusercam-decoded_sim_spatial_weight"
    # train_target_list =  "data/text_files/decoded_sim_captures_train.txt"
    # val_target_list = "data/text_files/decoded_sim_captures_val.txt"
    # learning_rate = 5e-4
    # fft_learning_rate = 5e-10
    batch_size = 5
    num_threads = 5
    decode_sim = True
    # val_train = True
    lambda_adversarial = 0.0
    use_spatial_weight = True
    multi = 10

def ours_diffusercam_decoded_sim_mulnew9_l1():
    exp_name = "fft-mulnew9-diffusercam-decoded_sim_spatial_weight_l1"
    train_target_list =  "data/text_files/decoded_sim_captures_train.txt"
    val_target_list = "data/text_files/decoded_sim_captures_val.txt"
    # learning_rate = 5e-4
    # fft_learning_rate = 5e-10
    batch_size = 5
    num_threads = 5
    # val_train = True
    lambda_adversarial = 0.0
    use_spatial_weight = True
    multi = 10
    lambda_l1 = 1.0
    lambda_image = 0


def ours_diffusercam_decoded_sim_mulnew9_unet_new():
    exp_name = "fft-mulnew9-diffusercam-decoded_sim_unet_new"
    train_target_list =  "data/text_files/decoded_sim_captures_train.txt"
    val_target_list = "data/text_files/decoded_sim_captures_val.txt"
    # learning_rate = 5e-4
    # fft_learning_rate = 5e-10
    batch_size = 5
    num_threads = 5
    model = "unet_new"
    pixelshuffle_ratio = 1
    # val_train = True
    lambda_adversarial = 0.0
    use_spatial_weight = True
    multi = 10
    

def ours_diffusercam_decoded_sim_mulnew9_no_pixelshuffle():
    exp_name = "fft-mulnew9-diffusercam-decoded_sim_spatial_weight_no_pixelshuffle"
    train_target_list =  "data/text_files/decoded_sim_captures_train.txt"
    val_target_list = "data/text_files/decoded_sim_captures_val.txt"
    # learning_rate = 5e-4
    # fft_learning_rate = 5e-10
    batch_size = 5
    num_threads = 5
    # val_train = True
    lambda_adversarial = 0.0
    use_spatial_weight = True
    multi = 10
    pixelshuffle_ratio = 1
    

def ours_diffusercam_decoded_sim_mulnew9_no_weight_update():
    exp_name = "fft-mulnew9-diffusercam-decoded_sim_spatial_weight_no_weight_update"
    train_target_list =  "data/text_files/decoded_sim_captures_train.txt"
    val_target_list = "data/text_files/decoded_sim_captures_val.txt"
    # learning_rate = 3e-4
    fft_learning_rate = 4e-10
    batch_size = 5
    num_threads = 5
    # val_train = True
    lambda_adversarial = 0.0
    use_spatial_weight = True
    multi = 10
    weight_update = False

def ours_diffusercam_decoded_sim_multi_ad_no_add_grad():
    exp_name = "fft-multi-diffusercam-1280-1408-decoded_sim_ad_no_add_grad"
    train_target_list =  "data/text_files/decoded_sim_captures_train.txt"
    val_target_list = "data/text_files/decoded_sim_captures_val.txt"
    # learning_rate = 3e-4
    # fft_learning_rate = 4e-10
    batch_size = 8
    num_threads = 4
    grad_lambda = 0.0
    # val_train = True
    lambda_adversarial = 0.6
    multi = 5
    learning_rate = 3e-4 
    fft_learning_rate = 4e-10 

def ours_diffusercam_decoded_sim_ad():
    exp_name = "fft-diffusercam-1280-1408-decoded_sim_ad"
    train_target_list =  "data/text_files/decoded_sim_captures_train.txt"
    val_target_list = "data/text_files/decoded_sim_captures_val.txt"
    learning_rate = 3e-4 
    fft_learning_rate = 4e-10 
    batch_size = 4
    num_threads = 4
    # lr_scheduler = "step"
    T_0 = 5
    pixelshuffle_ratio = 1
    # val_train = True
    lambda_adversarial = 0.6

def ours_diffusercam_decoded_sim_val_train():
    exp_name = "fft-diffusercam-1280-1408-decoded_sim_val_train"
    train_target_list =  "data/text_files/decoded_sim_captures_train.txt"
    val_target_list = "data/text_files/decoded_sim_captures_val.txt"
    # learning_rate = 3e-4
    # fft_learning_rate = 4e-10
    batch_size = 10
    num_threads = 5
    val_train = True
    lambda_adversarial = 0.0

def ours_diffusercam_val():
    exp_name = "ours-fft-diffusercam-1280-1408-val"
    # learning_rate = 3e-4
    # fft_learning_rate = 4e-10
    batch_size = 5
    num_threads = 5

def ours_diffusercam_single():
    exp_name = "ours-fft-diffusercam-1280-1408-single"
    batch_size = 5
    num_threads = 5

def ours_diffusercam_unet_64():
    exp_name = "ours-fft-diffusercam-1280-1408-unet-64"
    model = "unet-64-pixelshuffle-invert"


def ours_diffusercam_unet_32():
    exp_name = "ours-fft-diffusercam-1280-1408-unet-32"
    model = "unet-32-pixelshuffle-invert"


def ours_diffusercam_simulated():
    exp_name = "ours-fft-diffusercam-1280-1408-simulated"
    psf_mat = Path("data/phase_psf/sim_psf.npy")


def ours_meas_990_1254():
    exp_name = "ours-fft-diffusercam-990-1254"

    meas_crop_size_x = 990
    meas_crop_size_y = 1254

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def ours_meas_990_1254_simulated():
    exp_name = "ours-fft-diffusercam-990-1254-simulated"
    psf_mat = Path("data/phase_psf/sim_psf.npy")

    meas_crop_size_x = 990
    meas_crop_size_y = 1254

    use_mask = True


def ours_meas_864_1120():
    exp_name = "ours-fft-diffusercam-864-1120-big-mask"

    meas_crop_size_x = 864
    meas_crop_size_y = 1120

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def ours_meas_864_1120_simulated():
    exp_name = "ours-fft-diffusercam-864-1120-simulated"
    psf_mat = Path("data/phase_psf/sim_psf.npy")

    meas_crop_size_x = 864
    meas_crop_size_y = 1120

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_256x256_1280_1408.npy")


def ours_meas_608_864():
    exp_name = "ours-fft-diffusercam-608-864-big-mask"

    meas_crop_size_x = 608
    meas_crop_size_y = 864

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def ours_meas_608_864_simulated():
    exp_name = "ours-fft-diffusercam-608-864-simulated-big-mask"
    psf_mat = Path("data/phase_psf/sim_psf.npy")

    meas_crop_size_x = 608
    meas_crop_size_y = 864

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def ours_meas_512_640():
    exp_name = "ours-fft-diffusercam-512-640-big-mask"

    meas_crop_size_x = 512
    meas_crop_size_y = 640

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def ours_meas_512_640_simulated():
    exp_name = "ours-fft-diffusercam-512-640-simulated"
    psf_mat = Path("data/phase_psf/sim_psf.npy")

    meas_crop_size_x = 512
    meas_crop_size_y = 640

    use_mask = True


def ours_meas_400_400():
    exp_name = "ours-fft-diffusercam-400-400-big-mask"

    meas_crop_size_x = 400
    meas_crop_size_y = 400

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def ours_diffusercam_finetune_dualcam_1cap():
    exp_name = (
        "ours-fft-diffusercam-1280-1408-finetune-dualcam-fullG-1cap"
    )

    num_epochs = 63
    learning_rate = 3e-6

    static_val_image = "multicap_28.png"
    static_test_image = "test_set_Jan/output_cap_Image__2020-01-16__22-56-05.raw.png"
    image_dir = Path("data")
    text_file_dir = image_dir / "text_files"
    train_source_list = text_file_dir / "train_meas_1cap_indoor_dualcam.txt"
    train_target_list = text_file_dir / "train_webcam_indoor_dualcam.txt"
    val_source_list = text_file_dir / "val_meas_3cap_indoor_dualcam.txt"
    val_target_list = text_file_dir / "val_webcam_indoor_dualcam.txt"

    # Loss
    lambda_adversarial = 0.0
    lambda_contextual = 1
    lambda_perception = 0.0  # 0.006
    lambda_image = 0.0  # mse


def ours_meas_608_864_finetune_dualcam_1cap():
    exp_name = (
        "ours-fft-diffusercam-608-864-finetune-dualcam-fullG-1cap"
    )

    num_epochs = 127

    batch_size = 6
    learning_rate = 3e-6
    meas_crop_size_x = 608
    meas_crop_size_y = 864

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")

    static_val_image = "multicap_28.png"
    static_test_image = "test_set_Jan/output_cap_Image__2020-01-16__22-56-05.raw.png"
    image_dir = Path("data")
    text_file_dir = image_dir / "text_files"
    train_source_list = text_file_dir / "train_meas_1cap_indoor_dualcam.txt"
    train_target_list = text_file_dir / "train_webcam_indoor_dualcam.txt"
    val_source_list = text_file_dir / "val_meas_3cap_indoor_dualcam.txt"
    val_target_list = text_file_dir / "val_webcam_indoor_dualcam.txt"

    # Loss
    lambda_adversarial = 0.0
    lambda_contextual = 1
    lambda_perception = 0.0  # 0.006
    lambda_image = 0.0  # mse

    # Mask
    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def ours_meas_990_1254_finetune_dualcam_1cap():
    exp_name = (
        "ours-fft-diffusercam-990-1254-finetune-dualcam-fullG-1cap"
    )

    num_epochs = 127

    batch_size = 6
    learning_rate = 3e-6
    meas_crop_size_x = 990
    meas_crop_size_y = 1254

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")

    static_val_image = "multicap_28.png"
    static_test_image = "test_set_Jan/output_cap_Image__2020-01-16__22-56-05.raw.png"
    image_dir = Path("data")
    text_file_dir = image_dir / "text_files"
    train_source_list = text_file_dir / "train_meas_1cap_indoor_dualcam.txt"
    train_target_list = text_file_dir / "train_webcam_indoor_dualcam.txt"
    val_source_list = text_file_dir / "val_meas_3cap_indoor_dualcam.txt"
    val_target_list = text_file_dir / "val_webcam_indoor_dualcam.txt"

    # Loss
    lambda_adversarial = 0.0
    lambda_contextual = 1
    lambda_perception = 0.0  # 0.006
    lambda_image = 0.0  # mse

    # Mask
    use_mask = True
    mask_path = Path("data/phase_psf/_big_mask.npy")


def naive_meas_1280_1408():
    exp_name = "naive-fft-diffusercam-1280-1408"


def naive_meas_1280_1408_unet_64():
    exp_name = "naive-fft-diffusercam-1280-1408-unet-64"
    model = "unet-64-pixelshuffle-invert"


def naive_meas_1280_1408_unet_32():
    exp_name = "naive-fft-diffusercam-1280-1408-unet-32"
    model = "unet-32-pixelshuffle-invert"


def naive_meas_1280_1408_simulated():
    exp_name = "naive-fft-diffusercam-1280-1408-simulated"
    psf_mat = Path("data/phase_psf/sim_psf.npy")


def naive_meas_990_1254():
    exp_name = "naive-fft-diffusercam-990-1254-big-mask"

    meas_crop_size_x = 990
    meas_crop_size_y = 1254

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def naive_meas_990_1254_simulated():
    exp_name = "naive-fft-diffusercam-990-1254-simulated"
    psf_mat = Path("data/phase_psf/sim_psf.npy")

    meas_crop_size_x = 990
    meas_crop_size_y = 1254

    use_mask = True


def naive_meas_864_1120():
    exp_name = "naive-fft-diffusercam-864-1120-big-mask"

    meas_crop_size_x = 864
    meas_crop_size_y = 1120

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def naive_meas_864_1120_simulated():
    exp_name = "naive-fft-diffusercam-864-1120-simulated"
    psf_mat = Path("data/phase_psf/sim_psf.npy")

    meas_crop_size_x = 864
    meas_crop_size_y = 1120

    use_mask = True


def naive_meas_608_864():
    exp_name = "naive-fft-diffusercam-608-864-big-mask"

    meas_crop_size_x = 608
    meas_crop_size_y = 864

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def naive_meas_608_864_simulated():
    exp_name = "naive-fft-diffusercam-608-864-simulated-big-mask"

    meas_crop_size_x = 608
    meas_crop_size_y = 864

    psf_mat = Path("data/phase_psf/sim_psf.npy")

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def naive_meas_512_640():
    exp_name = "naive-fft-diffusercam-512-640-big-mask"

    meas_crop_size_x = 512
    meas_crop_size_y = 640

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def naive_meas_512_640_simulated():
    exp_name = "naive-fft-diffusercam-512-640-simulated"

    meas_crop_size_x = 512
    meas_crop_size_y = 640

    psf_mat = Path("data/phase_psf/sim_psf.npy")

    use_mask = True


def naive_meas_400_400():
    exp_name = "naive-fft-diffusercam-400-400-big-mask"

    meas_crop_size_x = 400
    meas_crop_size_y = 400

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")






def le_admm_meas_1280_1408_unet_64():
    exp_name = "le-admm-fft-diffusercam-1280-1408-unet-64"
    model = "unet-64-pixelshuffle-invert"


def le_admm_meas_1280_1408_unet_32():
    exp_name = "le-admm-fft-diffusercam-1280-1408-unet-32"
    model = "unet-32-pixelshuffle-invert"


def le_admm_meas_1280_1408_simulated():
    exp_name = "le-admm-fft-diffusercam-1280-1408-simulated"

    psf_mat = Path("data/phase_psf/sim_psf.npy")


def le_admm_meas_990_1254():
    exp_name = "le-admm-fft-diffusercam-990-1254"

    meas_crop_size_x = 990
    meas_crop_size_y = 1254

    normalise_admm_psf = True


def le_admm_meas_990_1254_simulated():
    exp_name = "le-admm-fft-diffusercam-990-1254-simulated"
    meas_crop_size_x = 990
    meas_crop_size_y = 1254

    normalise_admm_psf = True
    psf_mat = Path("data/phase_psf/sim_psf.npy")


def le_admm_meas_864_1120():
    exp_name = "le-admm-fft-diffusercam-864-1120"

    meas_crop_size_x = 840
    meas_crop_size_y = 1120

    normalise_admm_psf = True


def le_admm_meas_864_1120_simulated():
    exp_name = "le-admm-fft-diffusercam-864-1120-simulated"
    meas_crop_size_x = 840
    meas_crop_size_y = 1120

    normalise_admm_psf = True
    psf_mat = Path("data/phase_psf/sim_psf.npy")


def le_admm_meas_608_864():
    exp_name = "le-admm-fft-diffusercam-608-864"
    meas_crop_size_x = 608
    meas_crop_size_y = 864

    normalise_admm_psf = True


def le_admm_meas_608_864_simulated():
    exp_name = "le-admm-fft-diffusercam-608-864-simulated"
    meas_crop_size_x = 608
    meas_crop_size_y = 864

    normalise_admm_psf = True
    psf_mat = Path("data/phase_psf/sim_psf.npy")


def le_admm_meas_512_640():
    exp_name = "le-admm-fft-diffusercam-512-640"
    meas_crop_size_x = 512
    meas_crop_size_y = 640

    normalise_admm_psf = True


def le_admm_meas_512_640_simulated():
    exp_name = "le-admm-fft-diffusercam-512-640-simulated"
    meas_crop_size_x = 608
    meas_crop_size_y = 864

    normalise_admm_psf = True
    psf_mat = Path("data/phase_psf/sim_psf.npy")


def le_admm_meas_400_400():
    exp_name = "le-admm-fft-diffusercam-512-640"
    meas_crop_size_x = 400
    meas_crop_size_y = 400

    normalise_admm_psf = True


named_config_ll = [
    # Ours
    ours_diffusercam,
    ours_diffusercam_single,
    ours_diffusercam_val,
    ours_diffusercam_simulated,
    ours_meas_990_1254,
    ours_meas_990_1254_simulated,
    ours_meas_864_1120,
    ours_meas_864_1120_simulated,
    ours_meas_608_864,
    ours_meas_608_864_simulated,
    ours_meas_512_640,
    ours_meas_512_640_simulated,
    ours_meas_400_400,
    # Naive
    naive_meas_1280_1408,
    naive_meas_1280_1408_simulated,
    naive_meas_990_1254,
    naive_meas_990_1254_simulated,
    naive_meas_864_1120,
    naive_meas_864_1120_simulated,
    naive_meas_608_864,
    naive_meas_608_864_simulated,
    naive_meas_512_640,
    naive_meas_512_640_simulated,
    naive_meas_400_400,
    # Le ADMM
    le_admm_diffusercam,
    le_admm_meas_1280_1408_simulated,
    le_admm_meas_990_1254,
    le_admm_meas_990_1254_simulated,
    le_admm_meas_864_1120,
    le_admm_meas_864_1120_simulated,
    le_admm_meas_608_864,
    le_admm_meas_608_864_simulated,
    le_admm_meas_512_640,
    le_admm_meas_512_640_simulated,
    le_admm_meas_400_400,
    # Finetune
    ours_diffusercam_finetune_dualcam_1cap,
    ours_meas_608_864_finetune_dualcam_1cap,
    ours_meas_990_1254_finetune_dualcam_1cap,
    # Unet 64
    ours_diffusercam_unet_64,
    naive_meas_1280_1408_unet_64,
    le_admm_meas_1280_1408_unet_64,
    # Unet 32
    ours_diffusercam_unet_32,
    naive_meas_1280_1408_unet_32,
    le_admm_meas_1280_1408_unet_32,
    #decoded_sim
    ours_diffusercam_decoded_sim,
    ours_diffusercam_decoded_sim_val_train,
    ours_diffusercam_decoded_sim_ad,
    ours_diffusercam_decoded_sim_multi_ad_no_add_grad,
    #no decoded_sim
    ours_diffusercam_multi,
    ours_diffusercam_mulnew,
    ours_diffusercam_decoded_sim_mulnew9,
    ours_diffusercam_decoded_sim_mulnew9_no_weight_update,
    ours_diffusercam_decoded_sim_mulnew9_no_pixelshuffle,
    ours_diffusercam_decoded_sim_mulnew9_unet_new,
    ours_diffusercam_decoded_sim_mulnew9_l1,
    #concat_input
    ours_diffusercam_concat_input,
    ours_diffusercam_mulnew_concat_input,
    ours_diffusercam_mulnew_unet,
    ours_diffusercam_mulnew_unet_padding,
    ours_diffusercam_mulnew_unet_zero_conv,
    ours_diffusercam_mulnew_unet_padding_decode_sim
   
]


def initialise(ex):
    ex.config(base_config)
    for named_config in named_config_ll:
        ex.named_config(named_config)
    return ex

height = 270 * 4
width = 480 * 4
fft_args = {
    "psf_mat": Path("data/diffusercam/psf.tiff"),
    "psf_height": height,
    "psf_width": width,
    "psf_centre_x": height // 2,
    "psf_centre_y": width // 2,
    "psf_crop_size_x": height,
    "psf_crop_size_y": width,
    "meas_height": height,
    "meas_width": width,
    "meas_centre_x": height // 2,
    "meas_centre_y": width // 2,
    "meas_crop_size_x": height,
    "meas_crop_size_y": width,
    "pad_meas_mode": "replicate",
    # Change meas_crop_size_{x,y} to crop sensor meas. This will assume your sensor is smaller than the
    # measurement size. True measurement size is 1280x1408x4. Anything smaller than this requires padding of the
    # cropped measurement and then multiplying this with gaussian filtered rectangular box. For simplicity use the arguments
    # already set. Currently we are using full measurement. 
    "image_height": 270,
    "image_width": 480,
    "fft_gamma": 100,  # Gamma for Weiner init
    "use_mask": False,  # Use mask for cropped meas only
    "mask_path": Path("data/phase_psf/box_gaussian_1280_1408.npy"),
    # use Path("box_gaussian_1280_1408.npy") for controlled lighting
    # use Path("box_gaussian_1280_1408_big_mask.npy") for uncontrolled lighting
    "fft_requires_grad": False,
    "fft_epochs": 0,
    "concat_input": False,
} 

fft_args = SimpleNamespace(**fft_args)






if __name__ == "__main__":
    str_named_config_ll = [str(named_config) for named_config in named_config_ll]
    print("\n".join(str_named_config_ll))
