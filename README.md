# Image Deconvolution with Tikhonov Regularization

Mathematics for Imaging ans Signal Processing project

Link to the Web Application is [here](https://deconvolution-tikhonov.streamlit.app/)

## Overview

This repository explores the problem of image blurring and deblurring. It includes a Jupyter Notebook and a Streamlit web application, enabling users to interactively adjust regularization parameters and visualize their impact.

## Direct Problem: Blurring

First we need to obtain the blurred and noisy image (g), from the original one (f)

$$
g = Af + w
$$

$A$ is the convolution operator and $w$ is Gaussian noise sampled from $\mathcal{N}(\mu,\sigma^2)$

There are implemented the following blurring kernels:

- Gaussian Blur
- Linear Motion Blur
- Out-of-Focus Blur

## Inverse problem: Deblurring

### Deconvolution with $L^2$ Regularization

The solution of the following minimization problem:

$$
\min_f \| Af - g \|^2 + \mu \| f \|^2
$$

Is given by:

$$
\hat{f_\mu} = \frac{\overline{\hat{K}} \hat{g}}{|\hat{K}|^2 + \mu}
$$

### Deconvolution with $H^1$ Regularization

The solution of the following minimization problem:

$$
\min_f \| Af - g \|^2 + \mu \left( \| f \|^2 + \| \nabla f \|^2 \right)
$$

Is given by:

$$
\hat{f_\mu} = \frac{\overline{\hat{K}} \hat{g}}{|\hat{K}|^2 + \mu (1 + |\omega_1|^2 + |\omega_2|^2)}
$$

Where $\omega_1$ and $\omega_2$ are the spatial frequency variables.

### Deconvolution with Window Functions

We have implemented the following window functions:

- **Truncated Window** (Miller Regularization)

$$
\hat{W}_\mu(\omega) =
\begin{cases}
1, & \text{if } |\hat{K}(\omega)| > \sqrt{\mu}, \\
0, & \text{otherwise}.
\end{cases}
$$

- **Rectangular Window**

$$
\hat{W}_\Omega(\omega) =
\begin{cases}
1, & \text{if } |\omega| < \Omega, \\
0, & \text{otherwise}.
\end{cases}
$$

- **Triangular Window**

$$
\hat{W}_\Omega(\omega) =
\begin{cases}
1 - \frac{|\omega|}{\Omega}, & \text{if } |\omega| < \Omega, \\
0, & \text{otherwise}.
\end{cases}
$$

- **Generalized Hamming Window**

$$
\hat{W}_\Omega(\omega) =
\begin{cases}
\alpha + (1 - \alpha) \cos \left( \frac{\pi \omega}{\Omega} \right), & \text{if } |\omega| < \Omega, \\
0, & \text{otherwise}.
\end{cases}
$$

&nbsp;&nbsp;&nbsp; where $\alpha$ is a parameter (the default is $\alpha = 0.54$).

- **Gaussian Window**

$$
\hat{W}_\Omega(\omega) = \exp \left( - \frac{1}{2} \left(\frac{\omega}{\Omega}\right)^2 \right).
$$

From the window functions we get back the Fuorier Transform of the deconvolved image by applying the following formula:

$$
\hat{f_\Omega} = \hat{W}_\Omega \cdot \frac{\hat{g}}{\hat{K}}
$$

In general higher $\Omega$ values means that we are keeping also higher frequencies $\omega$, so we get sharper but noiser images. Instead $\mu$ is the regularization coefficient, the higher $\mu$ the higher the regularization so the smoother the function.

## Credits

- **Web Application**: Built using [Streamlit](https://streamlit.io/) to provide an interactive experience.
