import numpy as np
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2


################# KERNELS #################


def gaussian_kernel(size=15, sigma=3):
    """
    Generate a Gaussian kernel.
        - size: Size of the kernel (must be odd).
        - sigma: Standard deviation of the Gaussian.

        return: Normalized 2D Gaussian kernel.
    """
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, y)
    kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)  # Normalize the kernel to sum to 1


def motion_blur_kernel(size=15):
    """
    Create a linear motion blur kernel.
        - size: Size of the kernel (must be odd).
        - angle: Direction of motion in degrees (0 = horizontal).

        return: Motion blur kernel.
    """
    kernel = np.zeros((size, size))
    center = size // 2
    kernel[center, :] = 1.0

    return kernel / np.sum(kernel)


# Function to create a circular kernel for the out-of-focus blur
def out_of_focus_blur_kernel(D, size):
    # Create a meshgrid for the kernel (size x size)
    y, x = np.ogrid[-size // 2 : size // 2, -size // 2 : size // 2]

    # Calculate the distance from the center
    distance = x**2 + y**2

    # Generate the kernel based on the disk shape (circle of confusion)
    kernel = np.where(distance <= (D / 2) ** 2, 1, 0)

    return kernel / np.sum(kernel)


###########################################


def apply_blur_convolve2d(image, kernel):
    """
    Apply Gaussian blur to the image using directly the convolution.
    return: Blurred image.
    """
    blurred = convolve2d(image, kernel, mode="same", boundary="wrap")
    return blurred


def apply_blur_fft(image, kernel):
    """
    Apply Gaussian blur to the image using the FFT.
    return: Blurred image.
    """
    # Compute the 2D FFT of the image and the kernel
    image_fft = fft2(image)
    kernel_fft = fft2(kernel, s=image.shape)  # Pad the kernel to the image size

    # Perform the convolution in the frequency domain
    blurred_fft = image_fft * kernel_fft
    blurred = ifft2(blurred_fft).real

    blurred = np.clip(blurred, 0, 1)

    return blurred


def add_gaussian_noise(image, sigma=0.05):
    """
    Add Gaussian noise to the image.
        - image: Blurred image.
        - sigma: Standard deviation of the Gaussian.

        return: Noisy image.
    """
    np.random.seed(42)  # so we get always the same noise
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)  # Ensure pixel values are in [0, 1]
