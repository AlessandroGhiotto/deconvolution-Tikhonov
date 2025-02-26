import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2, fftshift
from code.blurring import *
from code.deblurring import *
from code.utilities import *
from functools import partial


# PART 1
def choose_image():
    cols = st.columns([1, 1])

    with cols[0]:
        img_str = st.selectbox(
            "Select Image",
            [
                "Upload Your Image",
                "Cat",
                "Dog",
                "Ducks",
                "Grasshopper",
                "Horse",
                "Rabbits",
                "Spider",
            ],
        )
        uploaded_img = None
        if img_str == "Upload Your Image":
            uploaded_img = st.file_uploader(
                "Upload an Image", type=["png", "jpg", "jpeg"]
            )

    with cols[1]:
        if img_str != "Upload Your Image":
            # Load predefined image
            img = Image.open(f"images/{img_str.lower()}.jpg").convert("L")
        elif uploaded_img:
            # Load user-uploaded image
            img = Image.open(uploaded_img).convert("L")
            img = resize_image(img)
        else:
            st.warning("Please upload an image or select a predefined one.")
            return None  # Stop execution if no image is selected

        # Convert image to numpy array and normalize
        img = np.array(img, dtype=np.float32)
        img = normalize_image(img)
        st.image(img, caption="Selected Image", use_container_width=True)

    return img  # Return the selected image as a NumPy array


# PART 2
# @st.cache_data(ttl=60)
def blur(img):
    cols = st.columns([1, 1])

    with cols[0]:
        # Choose blurring kernel
        blur_str = st.selectbox(
            "Select Blurring Kernel",
            [
                "Gaussian Blur",
                "Linear Motion Blur",
                "Out-of-Focus Blur",
            ],
        )

        # Pick the kernel based on the selected blurring type
        size = st.slider("Kernel Size", 3, 31, 15, 2)
        if blur_str == "Gaussian Blur":
            sigma = st.slider("Sigma", 1, 10, 3, 1)
            kernel = gaussian_kernel(size, sigma)
        elif blur_str == "Linear Motion Blur":
            kernel = motion_blur_kernel(size)
        elif blur_str == "Out-of-Focus Blur":
            D = st.slider("Diameter", 1, size, min(15, size), 2)
            kernel = out_of_focus_blur_kernel(D, size)
        sigma_noise = st.slider("SD for Gaussian Noise", 0.0, 0.1, 0.05, 0.01)

    with cols[1]:
        # Apply blurring and add noise
        blur_img = apply_blur_convolve2d(img, kernel)
        noisy_img = add_gaussian_noise(blur_img, sigma=sigma_noise)
        st.image(
            blur_img, caption="Blurred Image", use_container_width=True, clamp=True
        )
        st.image(
            noisy_img,
            caption="Blurred and Noisy Image",
            use_container_width=True,
            clamp=True,
        )

    return (
        add_gaussian_noise(
            apply_blur_fft(img, kernel), sigma=sigma_noise
        ),  # noisy_img_input
        kernel,  # kernel
    )


# PART 3
@st.fragment
def deblur(noisy_img_input, original_img, kernel):
    cols = st.columns([1, 1])

    with cols[0]:
        # Choose blurring kernel
        deblur_str = st.selectbox(
            "Select Deblurring Method",
            [
                "L2 Regularization",
                "H1 Regularization",
                "Truncated Window",
                "Rectangular Window",
                "Triangular Window",
                "Generalized Hamming Window",
                "Gaussian Window",
            ],
        )

        # Get params and deblur the image
        if deblur_str == "L2 Regularization":
            mu = st.slider(r"Select $\mu$ (power of 10) ", -4, 1, -1, 1)
            deblurred_img = deconvolve_L2(noisy_img_input, kernel, mu=10**mu)
        elif deblur_str == "H1 Regularization":
            mu = st.slider(r"Select $\mu$ (power of 10) ", -4, 1, -1, 1)
            deblurred_img = deconvolve_H1(noisy_img_input, kernel, mu=10**mu)
        elif deblur_str == "Truncated Window":
            mu = st.slider(r"$\mu$", 0.0, 1.0, 0.1, 0.01)
            deblurred_img = deconvolve_with_kernel_window(noisy_img_input, kernel, mu)
        elif deblur_str == "Rectangular Window":
            big_omega = st.slider(r"$\Omega$", 0.0, 0.3, 0.1, 0.01)
            deblurred_img = deconvolve_with_frequency_window(
                noisy_img_input, kernel, big_omega, rectangular_window
            )
        elif deblur_str == "Triangular Window":
            big_omega = st.slider(r"$\Omega$", 0.0, 0.3, 0.1, 0.01)
            deblurred_img = deconvolve_with_frequency_window(
                noisy_img_input, kernel, big_omega, triangular_window
            )
        elif deblur_str == "Generalized Hamming Window":
            big_omega = st.slider(r"$\Omega$", 0.0, 0.3, 0.1, 0.01)
            alpha = st.slider(r"$\alpha$", 0.0, 1.0, 0.54, 0.01)
            generalized_hamming_window_alpha = partial(
                generalized_hamming_window, alpha=alpha
            )
            deblurred_img = deconvolve_with_frequency_window(
                noisy_img_input,
                kernel,
                big_omega,
                generalized_hamming_window_alpha,
            )
        elif deblur_str == "Gaussian Window":
            big_omega = st.slider(r"$\Omega$", 0.0, 0.1, 0.06, 0.01)
            deblurred_img = deconvolve_with_frequency_window(
                noisy_img_input, kernel, big_omega, gaussian_window
            )

        # Compute PSNR
        if deblurred_img is not None:
            psnr_val = psnr(original_img, deblurred_img)

    with cols[1]:
        # Display deblurred image
        if deblurred_img is not None:
            deblurred_img_normalized = normalize_image(deblurred_img)
            st.image(
                deblurred_img_normalized,
                caption="Deblurred Image",
                use_container_width=True,
                clamp=True,
            )
            st.write(f"PSNR: {psnr_val:.2f} dB")
        else:
            st.warning("Please select a deblurring method.")


def main():
    st.title("Image Deconvolution with Tikhonov Regularization", anchor=False)
    st.markdown(
        """Author: &nbsp; Alessandro Ghiotto &nbsp;
        [![Personal Profile](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/AlessandroGhiotto) 
        """,
        unsafe_allow_html=True,
    )

    # Choose an image
    img = choose_image()

    if img is not None:
        # Apply blurring
        noisy_img_input, kernel = blur(img)

        # Deblur the image
        deblur(noisy_img_input, img, kernel)


if __name__ == "__main__":
    np.random.seed(42)
    st.set_page_config(
        page_title="MISP",
        page_icon=":camera:",
        menu_items={
            "Report a bug": "https://github.com/AlessandroGhiotto/brain-modeling/issues",
        },
    )
    main()
