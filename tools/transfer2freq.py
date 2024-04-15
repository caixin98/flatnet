#transfer the PSF to the frequency domain and compute the absolute square of H for visualisation.
import torch
import numpy as np
import cv2
# Load PSF and convert to PyTorch tensor
psf_path = '/root/caixin/flatnet/data/phase_psf/psf.npy'
psf = np.load(psf_path)
# psf_path = "/root/caixin/flatnet/phlatcam/psf_results/psf_sensor_sin_0.png"
# psf = cv2.imread(psf_path, cv2.IMREAD_GRAYSCALE)
psf_tensor = torch.tensor(psf, dtype=torch.float32)

# Compute the Fourier transform of the PSF
psf_fft = torch.fft.fft2(psf_tensor)

# Compute magnitude squared |H|^2 for visualization
magnitude_squared = torch.abs(psf_fft)**2

# Shift the zero frequency component to the center for better visualization
magnitude_squared_shifted = torch.fft.fftshift(magnitude_squared)

# Convert to numpy for plotting
magnitude_squared_shifted_np = magnitude_squared_shifted.numpy()

# 使用log1p转换幅度以提升可视化效果，并将数据范围归一化到0-255，适用于保存成图像格式
magnitude_squared_shifted_np = np.log1p(magnitude_squared_shifted_np)
# print(magnitude_squared_shifted_np.min(), magnitude_squared_shifted_np.max(), magnitude_squared_shifted_np.mean(), magnitude_squared_shifted_np.dtype)
magnitude_normalized = cv2.normalize(magnitude_squared_shifted_np, None, 0, 255, cv2.NORM_MINMAX)
magnitude_normalized_uint8 = np.uint8(magnitude_normalized)
# magnitude_normalized_uint8 = cv2.equalizeHist(magnitude_normalized_uint8)




# 保存图像
cv2.imwrite('/root/caixin/flatnet/data/phase_psf/magnitude_squared_1.png', magnitude_normalized_uint8) # 更新路径