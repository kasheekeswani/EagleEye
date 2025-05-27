# EagleEye Filter

🦅 **EagleEye Filter** — "Sharp on the edges. Soft on the noise."

## Overview

The EagleEye filter is a custom edge-preserving smoothing filter designed specifically for low-light images with high-frequency noise. It effectively reduces noise while maintaining important image details like edges and textures, making it ideal for enhancing noisy, low-light photographs.

The filter is inspired by bilateral filtering but optimized to better balance noise reduction and edge preservation through adaptive, dual-domain weighting.

---

## Features

- **Edge-aware smoothing:** Adapts smoothing strength based on local intensity differences.
- **Gaussian spatial weighting:** Prioritizes nearby pixels.
- **Adaptive filtering:** Dynamically adjusts filtering using local image statistics.
- **Noise suppression:** Efficiently reduces high-frequency noise (e.g., salt & pepper, Gaussian).
- **Detail preservation:** Retains textures and sharp edges.

---

## How It Works

The filter computes a weighted average for each pixel combining:

- A **spatial Gaussian weight** based on pixel distance.
- A **range weight** based on intensity differences to preserve edges.

The output pixel is normalized by the sum of weights to avoid brightness shifts.

Mathematically:

\[
I'(x,y) = \frac{1}{W} \sum_{i=-k}^k \sum_{j=-k}^k G(i,j) \cdot S(i,j) \cdot I(x+i, y+j)
\]

Where:

- \( G(i,j) = e^{-\frac{i^2 + j^2}{2\sigma_s^2}} \) is the spatial Gaussian kernel.
- \( S(i,j) = e^{-\frac{(I(x+i,y+j) - I(x,y))^2}{2\sigma_r^2}} \) is the range kernel based on intensity difference.
- \( W = \sum_{i=-k}^k \sum_{j=-k}^k G(i,j) \cdot S(i,j) \) is the normalization factor.

---

## Usage

### Requirements

- Python 3.6+
- OpenCV (`cv2`)
- NumPy
- Matplotlib

### Installation

```bash
pip install opencv-python numpy matplotlib
