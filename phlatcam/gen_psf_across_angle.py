# generate different PSFs across different angles [0, 30] degrees

from gen_phase import genPhaseMask, percentile_normalization
from propagate import prop2D
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_gradient_magnitude
from skimage.filters import sobel
from waveprop.rs import angular_spectrum_np, angular_spectrum, direct_integration
from waveprop.util import sample_points
import noise  # Uses the noise library for Perlin noise, install with pip install noise
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import fresnel
from waveprop.util import sample_points, ft2, ift2, _get_dtypes, zero_pad, crop

def fresnel_conv(u_in, wv, d1, dz, device=None, dtype=None, d2=None, pad=True):
    """
    Fresnel numerical computation (through convolution perspective) that gives
    control over output sampling but at a higher cost of two FFTs.

    Based off of Listing 6.5 of "Numerical Simulation of Optical Wave
    Propagation with Examples in MATLAB" (2010). Added zero-padding and support
    for PyTorch.

    NB: only works for square sampling, as non-square would result in different
    magnification factors.

    Parameters
    ----------
    u_in : :py:class:`~numpy.ndarray`
        Input amplitude distribution, [Ny, Nx].
    wv : float
        Wavelength [m].
    d1 : float
        Input sampling period for both x-dimension and y-dimension [m].
    d2 : float or list
        Desired output sampling period for both x-dimension and y-dimension [m].
    dz : float
        Propagation distance [m].
    pad : bool
        Whether or not to zero-pad to linearize circular convolution. If the
        original signal has enough padding, this may not be necessary.
    device : "cpu" or "gpu"
        If using PyTorch, required. Device on which to perform computations.
    dtype : float32 or float 64
        Data type to use.

    """
    if torch.is_tensor(u_in) or torch.is_tensor(dz):
        is_torch = True
    else:
        is_torch = False
    if is_torch:
        assert device is not None, "Set device for PyTorch"
        if torch.is_tensor(u_in):
            u_in = u_in.to(device)
        if torch.is_tensor(dz):
            dz = dz.to(device)
    assert isinstance(d1, float)
    if d2 is None:
        d2 = d1
    else:
        assert isinstance(d2, float)
    if dtype is None:
        dtype = u_in.dtype
    ctype, ctype_np = _get_dtypes(dtype, is_torch)

    if pad:
        N_orig = np.array(u_in.shape)
        u_in = zero_pad(u_in)
    N = np.array(u_in.shape)
    k = 2 * np.pi / wv

    # source coordinates
    x1, y1 = sample_points(N=N, delta=d1)
    r1sq = x1**2 + y1**2

    # source spatial frequencies
    df1 = 1 / (N * d1)
    fX, fY = sample_points(N=N, delta=df1)
    fsq = fX**2 + fY**2

    # scaling parameter
    m = d2 / d1

    # observation plane
    x2, y2 = sample_points(N=N, delta=d2)
    r2sq = x2**2 + y2**2

    # quadratic phase factors
    Q2 = np.exp(-1j * np.pi**2 * 2 * dz / m / k * fsq).astype(ctype_np)
    if is_torch:
        Q2 = torch.tensor(Q2, dtype=ctype).to(device)
    if m == 1:
        Q1 = 1
        Q3 = 1
    else:
        Q1 = np.exp(1j * k / 2 * (1 - m) / dz * r1sq).astype(ctype_np)
        Q3 = np.exp(1j * k / 2 * (m - 1) / (m * dz) * r2sq).astype(ctype_np)
        if is_torch:
            Q1 = torch.tensor(Q1, dtype=ctype).to(device)
            Q3 = torch.tensor(Q3, dtype=ctype).to(device)

    # propagated field
    u_out = Q3 * ift2(Q2 * ft2(Q1 * u_in / m, delta=d1), delta_f=df1)


    return u_out, x2, y2

phMm = np.load('results/phMm.npy')
lambd = 0.532
method = 'fp'
numIters = 20
zMS = 1869 / 6
pxSz = 0.4
for i in range(30):
    sin_phase_mask = np.zeros((phMm.shape[0], phMm.shape[1]))
    tan_phase_mask = np.zeros((phMm.shape[0], phMm.shape[1]))
    for j in range(sin_phase_mask.shape[1]):
        sin_phase_mask[:, j] = 2 * np.pi / lambd * np.sin(i / 180 * np.pi) * pxSz * j
        tan_phase_mask[:, j] = 2 * np.pi / lambd * np.tan(i / 180 * np.pi) * pxSz * j
    wavefront = np.exp(1j * (phMm + sin_phase_mask))
    psf_sensor, x_asm , y_asm = angular_spectrum_np(wavefront, lambd, pxSz, zMS, N_out = (phMm.shape[0] * 2, phMm.shape[1] * 2))
    psf_sensor = percentile_normalization(np.abs(psf_sensor)**2)
    # visualize the PSF at the sensor plane
    plt.imsave('psf_results/psf_sensor_sin_%d.png' % i, psf_sensor, cmap='gray')
    # print('hello')
    wavefront = np.exp(1j * (phMm + tan_phase_mask))
    psf_sensor, _ , _ = fresnel_conv(wavefront, lambd, float(pxSz), zMS)
    # psf_sensor, _ , _ = shifted_fresnel(wavefront, lambd, float(pxSz), zMS, out_shift=out_shift)
    psf_sensor = percentile_normalization(np.abs(psf_sensor)**2)
    # visualize the PSF at the sensor plane
    plt.imsave('psf_results/psf_sensor_tan_%d.png'%i, psf_sensor, cmap='gray')
    
    # break