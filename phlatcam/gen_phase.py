import numpy as np
import matplotlib.pyplot as plt
import os
from propagate import prop2D
def normalize_data(data):
    """将数据归一化到[0, 1]范围内"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def percentile_normalization(data, lower_percentile=0, upper_percentile=99.9):
    """
    对给定的数据进行百分位数归一化。
    
    参数:
    - data: 要归一化的数据，numpy数组格式。
    - lower_percentile: 下界百分位数（例如5表示5%）。
    - upper_percentile: 上界百分位数（例如95表示95%）。
    
    返回:
    - 归一化后的数据，同样是numpy数组格式。
    """

    # 计算下界和上界百分位数值
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)

    # 将数据限制在这个范围内，超出范围的值被设为边界值
    data_clipped = np.clip(data, lower_bound, upper_bound)

    # 对范围内的数据进行归一化
    normalized_data = (data_clipped - lower_bound) / (upper_bound - lower_bound)

    return normalized_data
  
def genPhaseMask(psf, lambd, pxSz, thickness, numIters, method, save_path='./results'):
    """
    Generate phase mask using iterative techniques and save the intermediary results.

    Parameters:
    psf : ndarray
        Input point spread function.
    lambd : float
        Wavelength in micrometers.
    pxSz : float
        Pixel size in micrometers.
    thickness : float
        Thickness in micrometers.
    numIters : int
        Number of iterations.
    method : str
        'as' (angular spectrum) or 'fp' (fresnel propagation).
    save_path : str
        Directory path to save the results.

    Returns:
    phM : ndarray
        Phase map [0, 2pi).
    Mm : ndarray
        Field at sensor plane.
    MsA : ndarray
        Intensity or PSF at the sensor plane.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    zMS = thickness
    Mamp = np.sqrt(psf)
    Ms = Mamp

    netLenXY = np.array(psf.shape) * pxSz

    for ii in range(1, numIters + 1):
        Mm = prop2D(Ms, netLenXY, lambd, -zMS, method)
        Mm = np.nan_to_num(Mm / np.abs(Mm), nan=0.0)

        phM = np.angle(Mm)
        phM[phM < 0] = 2 * np.pi + phM[phM < 0]
        
        if np.min(phM) > np.pi:
            phM -= np.pi
        
        plt.imsave(os.path.join(save_path, f'phM_iter_{ii}.png'), phM, cmap='jet')

        Ms = prop2D(Mm, netLenXY, lambd, zMS, 'as')
        MsA = np.abs(Ms)**2
        Ms = Mamp * Ms / np.sqrt(MsA)
        
        plt.imsave(os.path.join(save_path, f'MsA_iter_{ii}.png'), MsA, cmap='gray')

    # Optionally save the last MsA and phM images
    print(np.max(phM), np.min(phM), phM.shape)
    print(np.max(MsA), np.min(MsA), np.mean(MsA), MsA.shape)
    # print(MsA)

    plt.imsave(os.path.join(save_path, 'final_phM.png'), normalize_data(phM), cmap='jet')
    print(percentile_normalization(MsA).mean())
    plt.imsave(os.path.join(save_path, 'final_MsA.png'), percentile_normalization(MsA), cmap='gray')

    # Mm = Mm.astype(np.float32)
    MsA = MsA.astype(np.float32)
    phM = np.angle(Mm)
    phM[phM < 0] = 2 * np.pi + phM[phM < 0]

    return phM, Mm, MsA