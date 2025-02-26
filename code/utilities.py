import numpy as np
from scipy.fft import fft2, fftshift
from PIL import Image


def resize_image(image, max_size=512):
    """Resize the image to have a maximum size of max_size."""
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (
        (image - min_val) / (max_val - min_val)
        if max_val > min_val
        else np.zeros_like(image)
    )


def psnr(original, reconstructed):
    """Compute Peak Signal-to-Noise Ratio (PSNR)."""
    mse = np.mean((original - reconstructed) ** 2)
    max_pixel = np.max(original)
    return 20 * np.log10(max_pixel**2 / mse)


def get_magnitude_spectra(image, size=None):
    """
    Compute the magnitude spectra of the image.
        - image: Input image.
        - size: Size of the image (used to compute the magnitude spectra).
                if None, image.shape is used.
                it is needed when we have to compute the magnitude spectra of the kernel.

        return: Magnitude spectra of the image.
    """
    size = size if size is not None else image.shape

    # Compute the 2D FFT of the image
    img_ft = fft2(image, s=size)
    img_ft_shifted = fftshift(img_ft)  # Shift zero frequency to center

    # Compute magnitude spectra (log scale for better visualization)
    img_ft_magnitude = np.log(1 + np.abs(img_ft_shifted))
    return img_ft_magnitude
