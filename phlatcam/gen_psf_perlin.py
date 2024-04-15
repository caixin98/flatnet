# given an input psf, and a Camera thickness (mask-sensor distance or the light propagation distance),we want to infer the phase mask that will generate the desired PSF at the sensor plane.
from gen_phase import genPhaseMask, percentile_normalization
from propagate import prop2D
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_gradient_magnitude
from waveprop.rs import angular_spectrum_np, angular_spectrum, direct_integration
from waveprop.fresnel import fresnel_conv, shifted_fresnel
from waveprop.util import sample_points
import noise  # Uses the noise library for Perlin noise, install with pip install noise
import cv2
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



def free_space_impulse_response(k, x, y, z):
    # r = np.sqrt(x**2 + y**2 + z**2)
    # return 1 / (2 * np.pi) * np.exp(1j * k * r) / r * z / r * (1 / r - 1j * k)
    lambd = 2 * np.pi / k
    return np.exp(1j * k * z) / (1j * lambd * z) * np.exp(1j * k * (x**2 + y**2) / (2 * z))


def direct_integration_fp(wavefront, lambd, pxSz, zMS, x_asm, y_asm):
    # wavefront: the wavefront at the mask plane
    # lambd: the wavelength
    # pxSz: the pixel size
    # zMS: the mask-sensor distance
    # x_asm: the x coordinates at the sensor plane
    # y_asm: the y coordinates at the sensor plane
    # return the PSF at the sensor plane
    k = 2 * np.pi / lambd
    N = (wavefront.shape[0],wavefront.shape[1])
    x1, y1 = sample_points(N=N, delta=pxSz / 2)
    psf = np.zeros((len(x_asm), len(y_asm)), dtype=complex)
    for i, x in enumerate(x_asm):
        for j, y in enumerate(y_asm):
            psf[i, j] = np.sum(wavefront * free_space_impulse_response(k, x-x1, y-y1, zMS)) * pxSz**2
    return psf





def perlin2D(size, res):
    """
    Generate a 2D numpy array of perlin noise.
    Args:
        size: tuple of dimensions (width, height)
        res: tuple of resolutions for x and y, respectively
    Returns:
        2D numpy array of perlin noise.
    """
    def pnoise(x, y):
        #fix the random seed
        np.random.seed(100)
        # Adjust res for noise generation
        return noise.pnoise2(x / res[0], y / res[1], repeatx=size[0], repeaty=size[1])
    
    return np.array([[pnoise(x, y) for x in range(size[0])] for y in range(size[1])])

def resize_image(image_array, new_width, method=Image.NEAREST):
    """
    Resize the image to a new width while maintaining aspect ratio.
    Args:
        image_array: Input image array to resize.
        new_width: New width to resize to.
        method: Resampling method used by PIL. Default is Image.NEAREST.
    Returns:
        Resized image as a numpy array.
    """
    img = Image.fromarray(image_array)
    aspect_ratio = img.height / img.width
    new_height = int(aspect_ratio * new_width)
    img = img.resize((new_width, new_height), method)
    return np.array(img)

# Parameters
pName = 'perlin12_20_example'
minFeature = 3.2   # um Width of contour
pxSz = 0.4  # Placeholder for pixel size, adjust as needed

# Perlin noise generation
pattern_size = (150,150)  # pixels
In = perlin2D(pattern_size, (20, 20))


# Normalize
In = In - np.min(In)
In = In / np.max(In)
In = (In * 255).astype(np.uint8)
#visualize the edge map
import matplotlib.pyplot as plt
plt.imsave('results/edge_map.png', In, cmap='gray')
# Edge detection (using Sobel operator as an example, similar to Canny in performance for this purpose)
M = cv2.Canny(In, 40, 100, 2)
M = (M > 0).astype(float)  # Convert edges to binary

M = cv2.imread('../data/phase_psf/perlin_canny_unscaled.png', cv2.IMREAD_GRAYSCALE)
# print(M.shape)
# # Resize to get desired contour width (Assuming `pxSz` is provided accurately)
#center crop M for half the width and height
M = M[M.shape[0]//4:3*M.shape[0]//4,M.shape[1]//4:3*M.shape[1]//4]

new_width = int((minFeature / pxSz) * M.shape[1])  # New width based on minFeature and pixel size
M_resized = resize_image(M, new_width)
# padding the image to make it twice the size

psf = M_resized
print(psf.shape,psf.max(),psf.min())
# define the wavelength
lambd = 0.532
method = 'fp'
numIters = 20
zMS = 1869 / 6
# p = 12
# visualize the PSF
import matplotlib.pyplot as plt
plt.imsave('results/psf.png', psf, cmap='gray')

phMm, Mm, MsA = genPhaseMask(psf, lambd, pxSz, zMS, numIters, method)
# # # save the phase mask as npy file
np.save('results/phMm.npy', phMm)
# load the phase mask
phMm = np.load('results/phMm.npy')
print(phMm.shape, phMm.min(), phMm.max())
p = 6
#generate a phase mask: the (x, y) is 2 * pi / lambd * sin\theta * pxSz * y 
sin_phase_mask = np.zeros((psf.shape[0], psf.shape[1]))
tan_phase_mask = np.zeros((psf.shape[0], psf.shape[1]))
for j in range(sin_phase_mask.shape[1]):
    sin_phase_mask[:, j] = 2 * np.pi / lambd * np.sin(np.pi/p) * pxSz * j
    tan_phase_mask[:, j] = 2 * np.pi / lambd * np.tan(np.pi/p) * pxSz * j
# # transfer the phase mask to the range [0, 2pi)
sin_phase_mask = (sin_phase_mask) % (2 * np.pi) 
# visualize the new phase mask
plt.imsave('results/sin_phase_mask.png', sin_phase_mask, cmap='jet')

#given the phase mask, we can generate the PSF at the sensor plane
# generate the wavefront at the mask plane based on the phase mask
# wavefront_at_infinity = np.fft.fftshift(np.fft.fft2(wavefront))
netLenXY = np.array(psf.shape) * pxSz

# amplitude = np.abs(Mm)
# #visualize the amplitude of Mm
# plt.imsave('results/amplitude.png', amplitude, cmap='gray')

# wavefront = np.exp(1j * phMm) * amplitude
wavefront = np.exp(1j * phMm)
# generate the PSF at the sensor plane
psf_sensor = prop2D(wavefront, netLenXY, lambd, zMS, method)
psf_sensor = percentile_normalization(np.abs(psf_sensor)**2)
# visualize the PSF at the sensor plane
plt.imsave('results/psf_sensor.png', psf_sensor, cmap='gray')

psf_sensor, x_asm , y_asm = angular_spectrum_np(wavefront, lambd, pxSz, zMS, N_out = (psf.shape[0] * 2, psf.shape[1] * 2))
psf_sensor = percentile_normalization(np.abs(psf_sensor)**2)
# visualize the PSF at the sensor plane
plt.imsave('results/psf_sensor_as.png', psf_sensor, cmap='gray')

# psf_sensor = direct_integration(wavefront, lambd, pxSz, zMS, x_asm[0], y_asm) 
# psf_sensor = percentile_normalization(np.abs(psf_sensor)**2)
# # visualize the PSF at the sensor plane
# plt.imsave('results/psf_sensor_fp.png', psf_sensor, cmap='gray')

# phMm + sin_phase_mask
wavefront = np.exp(1j * (sin_phase_mask + phMm))
# generate the PSF at the sensor plane
psf_sensor = prop2D(wavefront, netLenXY, lambd, zMS, method = "fp")
psf_sensor = percentile_normalization(np.abs(psf_sensor)**2)
# visualize the PSF at the sensor plane
plt.imsave('results/psf_sensor_sin_%d.png'%p, psf_sensor, cmap='gray')

# using angular spectrum method
psf_sensor, x_asm , y_asm = angular_spectrum_np(wavefront, lambd, pxSz, zMS, N_out = (psf.shape[0] * 2, psf.shape[1] * 2))
psf_sensor = percentile_normalization(np.abs(psf_sensor)**2)
# visualize the PSF at the sensor 
print("hello!")
plt.imsave('results/psf_sensor_as_sin_%d.png'%p, psf_sensor, cmap='gray')

# # using fresnel_one_step method
# psf_sensor, x_asm , y_asm = fresnel_conv(wavefront, lambd, float(pxSz), zMS)
# psf_sensor = percentile_normalization(np.abs(psf_sensor)**2)
# # visualize the PSF at the sensor plane
# plt.imsave('results/psf_sensor_fn_sin_%d.png'%p, psf_sensor, cmap='gray')



# phMm + tan_phase_mask
wavefront = np.exp(1j * (phMm + tan_phase_mask))
# # generate the PSF at the sensor plane
# psf_sensor = prop2D(wavefront, netLenXY, lambd, zMS, "fp")
# psf_sensor = percentile_normalization(np.abs(psf_sensor)**2)
# # visualize the PSF at the sensor plane
# plt.imsave('results/psf_sensor_fp_tan_%d.png'%p, psf_sensor, cmap='gray')
#using angular spectrum method
out_shift = [0, -np.tan(np.pi/p) * zMS / pxSz]
print(out_shift)

psf_sensor, _ , _ = fresnel_conv(wavefront, lambd, float(pxSz), zMS)
print(psf_sensor.shape)
# psf_sensor, _ , _ = shifted_fresnel(wavefront, lambd, float(pxSz), zMS, out_shift=out_shift)
psf_sensor = percentile_normalization(np.abs(psf_sensor)**2)
# visualize the PSF at the sensor plane
plt.imsave('results/psf_sensor_fn_tan_%d.png'%p, psf_sensor, cmap='gray')

# # print(x_asm, y_asm)
# #using direct integration method
# psf_sensor = direct_integration(wavefront, lambd, pxSz, zMS, x_asm[0], y_asm)
# psf_sensor = percentile_normalization(np.abs(psf_sensor)**2)
# # visualize the PSF at the sensor plane
# plt.imsave('results/psf_sensor_fp_tan_%d.png'%p, psf_sensor, cmap='gray')

# if __name__ == '__main__':
#     pass