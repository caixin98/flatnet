# given an input psf, and a Camera thickness (mask-sensor distance or the light propagation distance),we want to infer the phase mask that will generate the desired PSF at the sensor plane.
from gen_phase import genPhaseMask, percentile_normalization
from propagate import prop2D
import numpy as np
import torch
import torch.nn.functional as F
psf_path = '../data/phase_psf/psf.npy'
# Define the input PSF
psf = np.load(psf_path)
sensor = dict(size = np.array([4.8e-6 * 1518, 4.8e-6 * 2012]))

def crop_and_padding(img, meas_crop_size_x=1280, meas_crop_size_y=1408, meas_centre_x=808, meas_centre_y=965, psf_height=1518, psf_width=2012, pad_meas_mode="replicate"):
    # crop
    img = torch.tensor(img)
    if meas_crop_size_x and meas_crop_size_y:
        crop_x = meas_centre_x - meas_crop_size_x // 2
        crop_y = meas_centre_y - meas_crop_size_y // 2

        # Replicate padding
        img = img[
            crop_x: crop_x + meas_crop_size_x,
            crop_y: crop_y + meas_crop_size_y,
            ]

        pad_x = psf_height - meas_crop_size_x
        pad_y = psf_width - meas_crop_size_y
        
        img = F.pad(
            img.permute(2, 0, 1).unsqueeze(0),
            (pad_y // 2, pad_y // 2, pad_x // 2, pad_x // 2),
            mode=pad_meas_mode,
        )

        img = img.squeeze(0).permute(1, 2, 0)
        # resize for half
    img = img.numpy()
    # img = cv2.resize(img.numpy(), (meas_crop_size_y // 2, meas_crop_size_x // 2))
    
    # img = img[..., None]
    print("img shape: ", img.shape)
    return img

# add last dimension
psf = psf[..., None]
# print(psf.shape)
psf = crop_and_padding(psf)
psf = psf[..., 0]
# define the wavelength
lambd = 0.532
method = 'as'
numIters = 20
zMS = 1900
pxSz = 2
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


