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
import noise  # Uses the noise library for Perlin noise, install with pip install noise

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
        np.random.seed(0)
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
minFeature = 12  # um Width of contour
pxSz = 2  # Placeholder for pixel size, adjust as needed

# Perlin noise generation
pattern_size = (300, 300)  # pixels
In = perlin2D(pattern_size, (20, 20))

# Normalize
In = In - np.min(In)
In = In / np.max(In)

# Edge detection (using Sobel operator as an example, similar to Canny in performance for this purpose)
M = sobel(In)
M = (M > 0).astype(float)  # Convert edges to binary

# Resize to get desired contour width (Assuming `pxSz` is provided accurately)
new_width = int((minFeature / pxSz) * M.shape[1] / M.shape[0])  # New width based on minFeature and pixel size
M_resized = resize_image(M, new_width)

psf = M_resized
# define the wavelength
lambd = 0.532
method = 'as'
numIters = 20
zMS = 1869
# visualize the PSF
import matplotlib.pyplot as plt
plt.imsave('results/psf.png', psf, cmap='gray')
#visualize the phase mask
phase_mask_path = "/root/caixin/flatnet/data/phase_images/n02165456_10030.png"
phase_mask = plt.imread(phase_mask_path)
#normalize the phase mask
phase_mask = phase_mask / np.max(phase_mask)
plt.imsave('results/phase_mask.png', phase_mask, cmap='jet')
phMm, Mm, MsA = genPhaseMask(psf, lambd, pxSz, zMS, numIters, method)
#generate a phase mask: the (x, y) is 2 * pi / lambd * sin\theta * pxSz * y 
new_phase_mask = np.zeros((phase_mask.shape[0], phase_mask.shape[1]))
for j in range(phase_mask.shape[1]):
    new_phase_mask[:, j] = 2 * np.pi / lambd * np.sin(np.pi/6) * pxSz * j
# transfer the phase mask to the range [0, 2pi)
new_phase_mask = new_phase_mask % (2 * np.pi)
# visualize the new phase mask
plt.imsave('results/new_phase_mask.png', new_phase_mask, cmap='jet')

#given the phase mask, we can generate the PSF at the sensor plane
# generate the wavefront at the mask plane based on the phase mask
# wavefront_at_infinity = np.fft.fftshift(np.fft.fft2(wavefront))
netLenXY = np.array(psf.shape) * pxSz

amplitude = np.abs(Mm)
#visualize the amplitude of Mm
plt.imsave('results/amplitude.png', amplitude, cmap='gray')

# wavefront = np.exp(1j * phMm) * amplitude
wavefront = np.exp(1j * phMm)
# generate the PSF at the sensor plane
psf_sensor = prop2D(wavefront, netLenXY, lambd, zMS, method)
psf_sensor = percentile_normalization(np.abs(psf_sensor)**2)
# visualize the PSF at the sensor plane
plt.imsave('results/psf_sensor.png', psf_sensor, cmap='gray')


