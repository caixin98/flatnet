import numpy as np

def prop2D(Uin, Lxy, lambda_, z, method='as'):
    """
    Propagation - transfer function approach in 2D with uniform sampling.
    
    Parameters:
    Uin : ndarray
        Source plane field.
    Lxy : tuple or list
        (Lx, Ly) source and observation plane side lengths.
    lambda_ : float
        Wavelength.
    z : float
        Propagation distance.
    method : str, optional
        'as' (angular spectrum) or 'fp' (fresnel propagation). Default is 'as'.
    
    Returns:
    Uout : ndarray
        Observation plane field.
    """
    print(Uin.shape)
    Ny, Nx = Uin.shape
    Lx, Ly = Lxy

    fs_x = Nx / Lx
    fs_y = Ny / Ly

    fx = fs_x * (np.arange(-Nx/2, Nx/2) / Nx)
    fy = fs_y * (np.arange(-Ny/2, Ny/2) / Ny)
    Fx, Fy = np.meshgrid(fx, fy)

    bp = np.sqrt(Fx**2 + Fy**2) < (1/lambda_)  # Medium's Bandpass

    # Angular spectrum
    if method.lower() == 'as':
        H = bp * np.exp(1j * 2 * np.pi * (z / lambda_) * bp * np.sqrt(1 - lambda_**2 * (Fx**2 + Fy**2)))
    
    # Fresnel
    if method.lower() == 'fp':
        H = bp * np.exp(1j * 2 * np.pi * (z / lambda_) * bp * (1 - 0.5 * lambda_**2 * (Fx**2 + Fy**2)))

    A0 = np.fft.fft2(Uin)
    A0 = np.fft.fftshift(A0)
    
    Az = A0 * H
    
    Az = np.fft.ifftshift(Az)
    Uout = np.fft.ifft2(Az)

    return Uout