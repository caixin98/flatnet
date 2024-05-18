# given an input psf, and a Camera thickness (mask-sensor distance or the light propagation distance),we want to infer the phase mask that will generate the desired PSF at the sensor plane.
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
from waveprop.fresnel import fresnel_conv
from waveprop.util import sample_points
import noise  # Uses the noise library for Perlin noise, install with pip install noise
import cv2
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
minFeature = 1.05  # um Width of contour
pxSz = 0.35  # Placeholder for pixel size, adjust as needed

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
new_width = int((minFeature / pxSz) * M.shape[1])  # New width based on minFeature and pixel size
M_resized = resize_image(M, new_width)
# padding the image to make it twice the size

psf = M_resized
print(psf.shape,psf.max(),psf.min())
# define the wavelength
lambd = 0.532
method = 'fp'
numIters = 20
zMS = 1869 / 8
# p = 12
# visualize the PSF
import matplotlib.pyplot as plt
plt.imsave('results/psf.png', psf, cmap='gray')


p = 13
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


# phMm + sin_phase_mask
wavefront = np.exp(1j * (sin_phase_mask))
# # generate the PSF at the sensor plane
# psf_sensor = prop2D(wavefront, netLenXY, lambd, zMS, method = "fp")
# psf_sensor = percentile_normalization(np.abs(psf_sensor)**2)
# # visualize the PSF at the sensor plane
# plt.imsave('results/psf_sensor_sin_%d.png'%p, psf_sensor, cmap='gray')

# using angular spectrum method
psf_sensor, x_asm , y_asm = angular_spectrum_np(wavefront, lambd, pxSz, zMS, N_out = (psf.shape[0] * 2, psf.shape[1] * 2)) 
psf_sensor = percentile_normalization(np.abs(psf_sensor)**2)
# visualize the PSF at the sensor plane
plt.imsave('results/psf_sensor_as_sin_%d.png'%p, psf_sensor, cmap='gray')

# using fresnel_one_step method
psf_sensor, x_asm , y_asm = fresnel_conv(wavefront, lambd, float(pxSz), zMS)
psf_sensor = percentile_normalization(np.abs(psf_sensor)**2)
# visualize the PSF at the sensor plane
plt.imsave('results/psf_sensor_fn_sin_%d.png'%p, psf_sensor, cmap='gray')

