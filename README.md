# Computer Vision Pipeline: Feature Detection and Matching

## Overview
This project implements a classical Computer Vision pipeline for interest point detection, local feature description, and feature matching from scratch. The system features a native C++ backend exposing REST API endpoints, paired with an interactive frontend interface to visualize algorithm outputs dynamically.

## Architecture

* **Backend (C++)**: High-performance implementation of core computer vision algorithms. Exposes an API using `httplib`. Images are passed encoded in base64.
* **Frontend (HTML/JS/CSS)**: An interactive web application that allows users to upload images, send them to the backend pipeline, and visualize the feature extraction and matching processes visually.

## Algorithm Specifications

### 1. Interest Point Detection (Harris & Shi-Tomasi / Lambda Method)
The backend supports two classical and reliable corner scoring mechanisms. Both start by identifying regions with significant intensity variation.
* **Preprocessing:** Calculates gradients internally using appropriate $3 \times 3$ Sobel kernels to ensure noise robustness.
* **Structure Tensor:** Computes $Ix^2$, $Iy^2$, and $Ix \cdot Iy$ and filters them using separable Gaussian blurs.
* **Response Calculation:**
  * **Harris:** Computes the corner response $R = \det(M) - k \cdot \operatorname{trace}(M)^2$.
  * **Shi-Tomasi (Lambda Method):** Computes $R = \min(\lambda_1, \lambda_2)$ by extracting the exact minimum eigenvalue of the structure tensor directly, making the detector more stable under affine variations.
* **Keypoint Extraction:** Applies thresholding followed by a $7 \times 7$ window Non-Maximal Suppression (NMS). The strongest keypoints (capped at 3,000) are extracted.

### 2. Local Feature Description (SIFT-Like Descriptor)
Each detected interest point is encoded into an invariant vector.
* **Scale & Rotation:** Interpolates gradient magnitudes and orientations surrounding the keypoint.
* **Spatial Grid:** Utilizes a $16 \times 16$ pixel neighborhood broken into $4 \times 4$ spatial blocks.
* **Orientation Binning:** Each spatial block computes an 8-bin orientation histogram, resulting in a 128-dimensional feature vector ($4 \times 4 \times 8 = 128$).
* **Normalization:** Applies the Lowe’s standard normalization: the vector is L2 scaled, threshold-clamped at a maximum value of 0.2 to limit the influence of extreme gradients, and then re-normalized to unit length.

### 3. Feature Matching (SSD & NCC)
The pipeline supports finding correspondences between two images using two distinct similarity metrics: Sum of Squared Differences (SSD) and Normalized Cross-Correlation (NCC).
* **Nearest Neighbor Search:** For each keypoint in Image 1, the optimal corresponding keypoint in Image 2 is located based on Euclidean distance (for SSD) or dot product similarity (for NCC).
* **Lowe's Ratio Test:** Filters ambiguous matches by analyzing the ratio of the best match metric to the second-best match metric. Matches that fail to breach a predefined uniqueness ratio threshold ($0.75$ for SSD, $0.80$ for NCC) are rejected.
* **Cross-checking Validation:** Employs a robust bidirectional filter. A match between Keypoint A (Image 1) and Keypoint B (Image 2) is only retained if a reverse search confirms A is also the top match for B.
* **Visualization:** Generates side-by-side comparative outputs drawing distinct, dynamically colored lines using Golden Angle hue distribution (137.5 degrees) for maximum linear separability.

## Project Structure

```
├── Backend/                 # C++ implementation
│   ├── include/             # Third-party libraries (httplib, json.hpp, stb)
│   ├── operations/          # Vision algorithms (Harris, SIFT, Matcher)
│   ├── CMakeLists.txt       # Build system configuration
│   ├── main.cpp             # API server entrypoint
│   └── run.bat              # Build and execution script
├── Frontend/                # UI implementation (HTML, custom CSS, JS)
└── Test_Cases/              # Sample images for testing features
```

## Setup and Execution

1. **Compile the Backend**:
   Navigate to the `Backend/` directory and build using CMake, or simply run the provided batch script on Windows:
   ```cmd
   cd Backend
   run.bat
   ```
   This will build the project and spin up the HTTP server over `localhost:8080`.

2. **Launch the Frontend**:
   Serve the `Frontend/` dictionary using any local web server. For example, using Python:
   ```cmd
   cd Frontend
   python -m http.server 3000
   ```
   Open `http://localhost:3000` (or double click `operations.html` if permitted by browser CORS policies) to access the operations dashboard.