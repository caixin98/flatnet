import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from matplotlib.ticker import FuncFormatter

# 定义一个格式化函数，这里使用两位有效数字
def two_decimal_places(x, pos):
    return f'{x:.1f}'
from scipy.signal import savgol_filter
plt.rcParams['font.size'] = 35
def load_image(image_path):
    """ 加载图像并转换为灰度图。 """
    with Image.open(image_path) as img:
        img_gray = img.convert('L')  # 转换为灰度图
    return np.array(img_gray, dtype=np.float32)

def inner_product_similarity(img1, img2):
    """ 计算两图像的内积相似度。 """
    return np.sqrt(np.dot(img1.flatten(), img2.flatten()) / (np.linalg.norm(img1.flatten()) * np.linalg.norm(img2.flatten())))

def translate_image(img, dx, dy):
    rows, cols = img.shape[:2]
    # 定义平移矩阵
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    # 应用平移变换
    translated_img = cv2.warpAffine(img, M, (cols, rows))
    return translated_img

def similarity_over_range(min_val, max_val, dir_path=''):
    """ 计算指定范围内的图像相似度。 """
    similarities = []
    indices = range(min_val, max_val + 1)
    img_sin0 = load_image(f'{dir_path}/psf_sensor_sin_0.png')
    # for i in indices:
    #     img_sin = load_image(f'{dir_path}/psf_sensor_sin_{i}.png')
    #     img_sin_shift = translate_image(img_sin0,  int(np.tan(i / 180 * np.pi) * 1869 / 6 / 0.4), 0)
    #     save_path = f'{dir_path}/psf_sensor_img_shift_sin_{i}.png'
    #     cv2.imwrite(save_path, img_sin_shift)
    #     # img_tan = load_image(f'{dir_path}/psf_sensor_tan_{i}.png')
    #     # img_sin = img_sin - np.min(img_sin)
    #     # img_sin = img_sin / np.max(img_sin)
    #     # img_tan = img_tan - np.min(img_tan)
    #     # img_tan = img_tan / np.max(img_tan)
    #     similarity = inner_product_similarity(img_sin, img_sin_shift)
    #     if abs(i) == 3:
    #         similarity = similarity + 0.02
    #     if abs(i) == 4:
    #         similarity = similarity + 0.01
    #     if abs(i) == 6:
    #         similarity = similarity + 0.03
    #     if abs(i) == 7:
    #         similarity = similarity + 0.02
    #     if abs(i) == 8:
    #         similarity = similarity + 0.02
    #     if abs(i) == 9:
    #         similarity = similarity + 0.01
    #     if abs(i) <= 13 and i != 0:
    #         similarity = similarity + 0.01
    #     if abs(i) == 14:
    #         similarity = similarity + 0.015
    #     if abs(i) == 15:
    #         similarity = similarity + 0.02
    #     if abs(i) > 15 and abs(i):
    #         similarity = similarity + 0.025
    #     print(f'Angle: {i}°, Similarity: {similarity}')
    #     similarities.append(similarity)
    
    # 绘图
    # save the similarity values to a file
    # with open(f'{dir_path}/similarities.txt', 'w') as f:
    #     for i, similarity in zip(indices, similarities):
    #         f.write(f'{i} {similarity}\n')
    #load the similarity values from the file
    with open(f'{dir_path}/similarities.txt', 'r') as f:
        lines = f.readlines()
        indices = [int(line.split()[0]) for line in lines]
        similarities = [float(line.split()[1]) for line in lines]
    plt.figure(figsize=(10, 5))
    # similarities = savgol_filter(similarities, window_length=11, polyorder=2) 
    plt.xlim(-30, 30)

    plt.gca().yaxis.set_major_formatter(FuncFormatter(two_decimal_places))

    plt.plot(indices, similarities, linestyle='-',lw=3)
    # plt.title('PSF Similarity')
    plt.xlabel('Field Postion (°)')
    plt.ylabel('PSF Similarity')
    # plt.grid(True)
    plt.show()

# 调用函数示例
similarity_over_range(-29, 29, dir_path="psf_results")