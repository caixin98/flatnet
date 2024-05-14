"""
Get model
"""
from models.fftlayer import FFTLayer
from models.multi_fftlayer import MultiFFTLayer
from models.multi_fftlayer_diff import MultiFFTLayer_diff
from models.fftlayer_diff import FFTLayer_diff

from models.multi_fftlayer_new import MultiFFTLayer_new
from models.multi_fftlayer_shift import MultiFFTLayer_shift

from models.admm.admm_first_block import ADMM_Net as ADMM

from models.unet_128 import Unet as Unet_128
from models.unet_64 import Unet as Unet_64
from models.unet_32 import Unet as Unet_32
from models.cunet_128 import Unet as CUnet_128
from models.unet_new import Unet as Unet_new
from models.unet import UNet270480

from models.discriminator import Discriminator


def model(args):
    is_admm = "admm" in args.exp_name
    is_multi = "multi" in args.exp_name
    is_mulnew = "mulnew" in args.exp_name
    is_multi_shift = "shift" in args.exp_name and is_multi
    is_diff = "diff" in args.exp_name
    if is_multi:
        if not is_diff:
            in_c = 4 * args.multi
            Inversion = MultiFFTLayer
        else:
            in_c = 3 * args.multi
            Inversion = MultiFFTLayer_diff
    elif is_multi_shift:
        in_c = 4 * args.multi
        Inversion = MultiFFTLayer_shift
    elif is_admm:
        in_c = 3
        Inversion = ADMM
    elif is_mulnew:
        if not is_diff:
            in_c = 4 * 2
            Inversion = MultiFFTLayer_new
        else:
            in_c = 3 * 2
            Inversion = MultiFFTLayer_diff
            if args.concat_input:
                in_c = 9

    else:
        if not is_diff:
            in_c = 4
            Inversion = FFTLayer

        else:
            in_c = 3
            Inversion = FFTLayer_diff
            if args.concat_input:
                in_c = 6
    

    if args.model == "unet-128-pixelshuffle-invert":
        return Unet_128(args, in_c=in_c), Inversion(args), Discriminator(args)
    elif args.model == "unet-64-pixelshuffle-invert":
        return Unet_64(args, in_c=in_c), Inversion(args), Discriminator(args)
    elif args.model == "unet-32-pixelshuffle-invert":
        return Unet_32(args, in_c=in_c), Inversion(args), Discriminator(args)
    elif args.model == "cunet-128-pixelshuffle-invert":
        return CUnet_128(args, in_c=in_c), Inversion(args), Discriminator(args)
    elif args.model == "unet_new":
        return Unet_new(args, in_c=in_c), Inversion(args), Discriminator(args)
    elif args.model == "UNet270480":
        return UNet270480(args, in_c=in_c), Inversion(args), Discriminator(args)
    
