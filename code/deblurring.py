import numpy as np
from scipy.fft import fft2, ifft2


def deconvolve_L2(img, kernel, mu):
    """
    Perform L2 regularization deconvolution

        - input_img: Noisy image.
        - kernel: Blur kernel.
        - mu: Regularization parameter.

        return: Deconvolved image.
    """
    # fft2: compute the fourier transform
    g_hat = fft2(img)
    k_hat = fft2(kernel, s=img.shape)  # s=img.shape pad kernel to match image size
    f_hat = np.conj(k_hat) * g_hat / (np.abs(k_hat) ** 2 + mu)

    # convert the deconvolved image back to the spatial domain
    deconvolved_img = ifft2(f_hat).real
    deconvolved_img = np.clip(deconvolved_img, 0, 1)

    return deconvolved_img


# H1-regularized deconvolution function
def deconvolve_H1(img, kernel, mu):
    """
    Perform H1 regularization deconvolution.

        - img: Noisy image.
        - kernel: Blur kernel.
        - mu: Regularization parameter.

        return: Deconvolved image.
    """
    g_hat = fft2(img)
    k_hat = fft2(kernel, s=img.shape)

    # Get spatial frequency grid
    H, W = img.shape
    omega1, omega2 = np.meshgrid(np.fft.fftfreq(W), np.fft.fftfreq(H))

    # compute the deconvolved img (in frequency domain)
    denominator = np.abs(k_hat) ** 2 + mu * (1 + omega1**2 + omega2**2)
    f_hat = (np.conj(k_hat) * g_hat) / denominator

    # Convert deconvolved image back to spatial domain
    deconvolved_img = ifft2(f_hat).real
    deconvolved_img = np.clip(deconvolved_img, 0, 1)
    return deconvolved_img


################# WINDOW FUNCTIONS #################
# The truncated window (Miller Regularization) is done separately from the other window functions
# This window function take as input the k_hat, not the omegas


def truncated_window_function(K_hat, mu):
    """Truncated window function"""
    # Keeps frequencies where |K_hat| > sqrt(mu)
    W = np.zeros_like(K_hat)
    W[np.abs(K_hat) > np.sqrt(mu)] = 1
    return W


def deconvolve_with_kernel_window(g, K, mu, window_function=truncated_window_function):
    """
    Performs deconvolution using the window function.
    This method is used for window functions that depend on the kernel K_hat.
        - g: Noisy image.
        - K: Blur kernel.
        - mu: Regularization parameter.
        - window_function: Window function to apply, this is a function of the k_hat.

        return: Deconvolved image.
    """
    g_hat = fft2(g)
    K_hat = fft2(K, s=g.shape)

    # Apply the truncated window function
    W = window_function(K_hat, mu)
    f_hat = W * (g_hat / (K_hat + 1e-8))  # Avoid division by zero
    f_reconstructed = ifft2(f_hat).real
    return f_reconstructed


### OTHER WINDOW FUNCTIONS


def rectangular_window(omega1, omega2, big_omega):
    """Rectangular window"""
    omega_mag = np.sqrt(omega1**2 + omega2**2)  # Compute |omega|
    W = (omega_mag < big_omega).astype(float)  # 1 if |omega| < big_omega, else 0
    return W


def triangular_window(omega1, omega2, big_omega):
    """Triangular window"""
    omega_mag = np.sqrt(omega1**2 + omega2**2)
    W = np.maximum(1 - omega_mag / big_omega, 0)
    return W


def generalized_hamming_window(omega1, omega2, big_omega, alpha=0.54):
    """Generalized Hamming window"""
    omega_mag = np.sqrt(omega1**2 + omega2**2)  # Compute |omega|

    W = np.zeros_like(omega_mag)
    mask = omega_mag < big_omega  # 1 if |omega| < big_omega, else 0

    # for the values satisfying the condition we apply the formula (the rest will remain 0)
    W[mask] = alpha + (1 - alpha) * np.cos(np.pi * omega_mag[mask] / big_omega)
    return W


def gaussian_window(omega1, omega2, big_omega):
    """Gaussian window function"""
    omega_mag = np.sqrt(omega1**2 + omega2**2)
    W = np.exp(-(omega_mag**2) / (2 * big_omega**2))
    return W


# -------------------- General Deconvolution Method --------------------


def compute_frequency_grid(shape):
    """Computes the frequency grid (omega1, omega2) for given image shape."""
    H, W = shape
    # omega1: Frequency values along height
    # omega2: Frequency values along width
    omega1, omega2 = np.meshgrid(np.fft.fftfreq(W), np.fft.fftfreq(H))
    return omega1, omega2


def deconvolve_with_frequency_window(g, K, big_omega, window_function):
    """
    Performs deconvolution using a frequency-based window function.

    Parameters:
        - g: Observed blurred image (spatial domain)
        - K: Blurring kernel (spatial domain)
        - big_omega: Cutoff frequency parameter
        - window_function: window_function(omega1, omega2, big_omega) -> W

    Returns: f_reconstructed, Regularized reconstructed image
    """
    g_hat = fft2(g)
    K_hat = fft2(K, s=g.shape)

    # Compute frequency grid
    omega1, omega2 = compute_frequency_grid(g.shape)

    # Apply the selected window function in frequency domain
    W = window_function(omega1, omega2, big_omega)

    f_hat = W * (g_hat / (K_hat + 1e-8))
    f_reconstructed = ifft2(f_hat).real
    return f_reconstructed
