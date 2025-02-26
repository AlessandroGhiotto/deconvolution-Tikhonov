import streamlit as st
from PIL import Image
import numpy as np
from code.blurring import *
from code.deblurring import *
from code.utilities import *
from functools import partial


# PART 1
def choose_image():
    st.write("### Step 1: Choose Image")
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
        uploaded_img = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

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

    return img  # Return the selected image as a NumPy array


# PART 2
def blur(img):
    cols = st.columns(2)

    with cols[0]:
        st.write("### Step 2: Apply Blurring")
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
        # Apply blurring
        blur_img = apply_blur_convolve2d(img, kernel)
        blur_img_input = apply_blur_fft(img, kernel)

    with cols[1]:
        st.write("### Step 3: Add Noise")
        # Add noise
        sigma_noise = st.slider("SD for Gaussian Noise", 0.0, 0.3, 0.05, 0.01)
        noisy_img = add_gaussian_noise(blur_img, sigma=sigma_noise)
        # noisy_img_input is the one used for the deblurring
        # it is warped in the corner, but is fixed in the deblurring
        noisy_img_input = add_gaussian_noise(blur_img_input, sigma=sigma_noise)

    return blur_img, noisy_img, noisy_img_input, kernel


# PART 4
### NOTE
# Insane trick with the fragment, the cell given in the input is where I will plot the image
# so I can update in the fragment things in the layout that are outside of where I run the fragmnet iself
# I can just pass the cell, and greate there a new container and update it.
@st.fragment
def deblur(noisy_img_input, original_img, kernel, cell):
    st.write("### Step 4: Deblur Image")
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
        reconstructed_img = deconvolve_L2(noisy_img_input, kernel, mu=10**mu)
    elif deblur_str == "H1 Regularization":
        mu = st.slider(r"Select $\mu$ (power of 10) ", -4, 1, -1, 1)
        reconstructed_img = deconvolve_H1(noisy_img_input, kernel, mu=10**mu)
    elif deblur_str == "Truncated Window":
        mu = st.slider(r"$\mu$", 0.0, 1.0, 0.1, 0.01)
        reconstructed_img = deconvolve_with_kernel_window(noisy_img_input, kernel, mu)
    elif deblur_str == "Rectangular Window":
        big_omega = st.slider(r"$\Omega$", 0.01, 0.3, 0.1, 0.01)
        reconstructed_img = deconvolve_with_frequency_window(
            noisy_img_input, kernel, big_omega, rectangular_window
        )
    elif deblur_str == "Triangular Window":
        big_omega = st.slider(r"$\Omega$", 0.01, 0.3, 0.1, 0.01)
        reconstructed_img = deconvolve_with_frequency_window(
            noisy_img_input, kernel, big_omega, triangular_window
        )
    elif deblur_str == "Generalized Hamming Window":
        big_omega = st.slider(r"$\Omega$", 0.01, 0.3, 0.1, 0.01)
        alpha = st.slider(r"$\alpha$", 0.0, 1.0, 0.54, 0.01)
        generalized_hamming_window_alpha = partial(
            generalized_hamming_window, alpha=alpha
        )
        reconstructed_img = deconvolve_with_frequency_window(
            noisy_img_input,
            kernel,
            big_omega,
            generalized_hamming_window_alpha,
        )
    elif deblur_str == "Gaussian Window":
        big_omega = st.slider(r"$\Omega$", 0.01, 0.1, 0.06, 0.01)
        reconstructed_img = deconvolve_with_frequency_window(
            noisy_img_input, kernel, big_omega, gaussian_window
        )

    with cell.container():
        # Compute PSNR and show the deblurred image
        psnr_val = psnr(original_img, reconstructed_img)
        reconstructed_img_normalized = normalize_image(reconstructed_img)
        st.image(
            reconstructed_img_normalized,
            caption="Deblurred Image",
            use_container_width=True,
        )
        st.write(f"PSNR: {psnr_val:.2f} dB")

    return reconstructed_img


def main():
    st.title("Image Deconvolution with Tikhonov Regularization", anchor=False)
    st.markdown(
        """Author: &nbsp; Alessandro Ghiotto &nbsp;
        [![Personal Profile](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/AlessandroGhiotto) 
        """,
        unsafe_allow_html=True,
    )

    row1 = st.columns([1, 2, 1])
    row2 = st.columns(4)

    noisy_img_input = None
    kernel = None
    reconstructed_img = None

    # step 1
    with row1[0]:
        img = choose_image()
    # step 2 and 3
    with row1[1]:
        if img is not None:
            # Apply blurring and add noise
            blur_img, noisy_img, noisy_img_input, kernel = blur(img)
    # step 4
    with row1[2]:
        if noisy_img_input is not None:
            # Deblur the image
            reconstructed_img = deblur(noisy_img_input, img, kernel, row2[3].empty())

    # plot in the row below the images so that are all aligned
    if reconstructed_img is not None:
        with row2[0]:
            st.image(img, caption="Selected Image", use_container_width=True)
        with row2[1]:
            st.image(
                blur_img, caption="Blurred Image", use_container_width=True, clamp=True
            )
        with row2[2]:
            st.image(
                noisy_img,
                caption="Blurred and Noisy Image",
                use_container_width=True,
            )
        with row2[3]:
            # This is where the deblurred image will be displayed
            # It is updated in the deblur fragment
            pass


if __name__ == "__main__":
    np.random.seed(42)
    st.set_page_config(
        page_title="MISP",
        page_icon=":camera:",
        menu_items={
            "Report a bug": "https://github.com/AlessandroGhiotto/deconvolution-Tikhonov/issues",
        },
        layout="wide",
    )
    main()
