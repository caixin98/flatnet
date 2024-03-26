"""
Get model
"""
from models.fftlayer import FFTLayer
from models.multi_fftlayer import MultiFFTLayer
from models.multi_fftlayer_new import MultiFFTLayer_new

from models.admm.admm_first_block import ADMM_Net as ADMM

from models.unet_128 import Unet as Unet_128
from models.unet_64 import Unet as Unet_64
from models.unet_32 import Unet as Unet_32
from models.cunet_128 import Unet as CUnet_128



from models.discriminator import Discriminator


def model(args):
    is_admm = "admm" in args.exp_name
    is_multi = "multi" in args.exp_name
    is_mulnew = "mulnew" in args.exp_name
    if is_multi:
        in_c = 4 * args.multi
        Inversion = MultiFFTLayer
    elif is_admm:
        in_c = 3
        Inversion = ADMM
    elif is_mulnew:
        in_c = 4
        Inversion = MultiFFTLayer_new
    else:
        in_c = 4
        Inversion = FFTLayer
    

    if args.model == "unet-128-pixelshuffle-invert":
        return Unet_128(args, in_c=in_c), Inversion(args), Discriminator(args)

    elif args.model == "unet-64-pixelshuffle-invert":
        return Unet_64(args, in_c=in_c), Inversion(args), Discriminator(args)

    elif args.model == "unet-32-pixelshuffle-invert":
        return Unet_32(args, in_c=in_c), Inversion(args), Discriminator(args)
    elif args.model == "cunet-128-pixelshuffle-invert":
        return CUnet_128(args, in_c=in_c), Inversion(args), Discriminator(args)
