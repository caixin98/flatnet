import numpy as np
import matplotlib.pyplot as plt

# 设置参数
lambda_ = 633e-9  # 光波波长, 单位为米
k = 2 * np.pi / lambda_  # 波数
theta_x = np.deg2rad(5)  # x方向入射角度，转换为弧度
theta_y = np.deg2rad(0)  # y方向入射角度，转换为弧度
N = 1024  # 模拟区域的点数
L = 0.002  # 模拟区域的物理大小，单位为米（这将决定空间分辨率）
dx = L / N  # 空间分辨率
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# 生成线性相位项，以模拟倾斜入射
linear_phase = np.exp(1j * (k * np.sin(theta_x) * X + k * np.sin(theta_y) * Y))

# 假设相位mask，这里我们模拟一个简单的相位跃迁
mask = np.exp(1j * np.pi * (X > 0))  # X > 0的区域相位增加π

# 波前通过相位mask后的情况
wavefront = linear_phase * mask

# 使用傅立叶变换计算远场衍射模式
psf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wavefront)))

# 显示结果
plt.imshow(np.abs(psf)**2, cmap='viridis', extent=(-L/2, L/2, -L/2, L/2))
plt.colorbar()
plt.title('Intensity in the Far Field')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()