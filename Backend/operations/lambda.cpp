#include "lambda.h"
#include <cmath>
#include <algorithm>
#include <queue>
#include <utility>
#include <chrono>

static std::vector<std::vector<float>> compute_elementwise_product(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b);
static std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> compute_gradients(const ImageData& gray);
static std::vector<std::vector<float>> gaussian_blur(const std::vector<std::vector<float>>& img, float sigma);

static std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> compute_gradients(const ImageData& gray) {
    int w = gray.width, h = gray.height;
    std::vector<std::vector<float>> Ix(h, std::vector<float>(w, 0));
    std::vector<std::vector<float>> Iy(h, std::vector<float>(w, 0));
    
    // Sobel operators
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            Ix[y][x] = (gray.data[(y) * w + (x+1)] - gray.data[(y) * w + (x-1)]) / 2.0f;
            Iy[y][x] = (gray.data[(y+1) * w + x] - gray.data[(y-1) * w + x]) / 2.0f;
        }
    }
    return std::make_pair(Ix, Iy);
}

static std::vector<std::vector<float>> gaussian_blur(const std::vector<std::vector<float>>& img, float sigma) {
    int h = img.size(), w = img[0].size();
    int kernel_size = static_cast<int>(6 * sigma) | 1;
    std::vector<float> kernel(kernel_size);
    float sum = 0;
    
    for (int i = 0; i < kernel_size; i++) {
        int x = i - kernel_size / 2;
        kernel[i] = exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    for (float& k : kernel) k /= sum;
    
    std::vector<std::vector<float>> temp(h, std::vector<float>(w, 0));
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float val = 0;
            for (int kx = 0; kx < kernel_size; kx++) {
                int sx = x + kx - kernel_size / 2;
                if (sx >= 0 && sx < w) {
                    val += img[y][sx] * kernel[kx];
                }
            }
            temp[y][x] = val;
        }
    }
    
    std::vector<std::vector<float>> result(h, std::vector<float>(w, 0));
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float val = 0;
            for (int ky = 0; ky < kernel_size; ky++) {
                int sy = y + ky - kernel_size / 2;
                if (sy >= 0 && sy < h) {
                    val += temp[sy][x] * kernel[ky];
                }
            }
            result[y][x] = val;
        }
    }
    return result;
}

ShiTomasiResult detect_shi_tomasi(const ImageData& img, float k, int threshold, int nms_size) {
    auto start = std::chrono::high_resolution_clock::now();
    
    ImageData gray = img.channels > 1 ? grayscale(img) : img;
    auto [Ix, Iy] = compute_gradients(gray);
    
    // Compute structure tensor components with Gaussian blur
    auto Ixx = gaussian_blur(compute_elementwise_product(Ix, Ix), 1.5f);
    auto Iyy = gaussian_blur(compute_elementwise_product(Iy, Iy), 1.5f);
    auto Ixy = gaussian_blur(compute_elementwise_product(Ix, Iy), 1.5f);
    
    int w = gray.width, h = gray.height;
    std::vector<std::vector<float>> R(h, std::vector<float>(w, 0));
    
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float det = Ixx[y][x] * Iyy[y][x] - Ixy[y][x] * Ixy[y][x];
            float trace = Ixx[y][x] + Iyy[y][x];
            
            // Shi-Tomasi operator: R = lambda_min
            // lambda = (trace +/- sqrt(trace^2 - 4*det)) / 2
            float discriminant = trace * trace - 4 * det;
            if (discriminant < 0) discriminant = 0; // Avoid NaN due to floating point inaccuracies
            R[y][x] = (trace - std::sqrt(discriminant)) / 2.0f;
        }
    }
    
    // Non-maximum suppression
    std::vector<Keypoint> keypoints;
    int nms_radius = nms_size / 2;
    
    for (int y = nms_radius; y < h - nms_radius; y++) {
        for (int x = nms_radius; x < w - nms_radius; x++) {
            float val = R[y][x];
            if (val < threshold) continue;
            
            bool is_max = true;
            for (int dy = -nms_radius; dy <= nms_radius && is_max; dy++) {
                for (int dx = -nms_radius; dx <= nms_radius; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    if (R[y + dy][x + dx] >= val) {
                        is_max = false;
                        break;
                    }
                }
            }
            
            if (is_max) {
                Keypoint kp;
                kp.x = static_cast<float>(x);
                kp.y = static_cast<float>(y);
                kp.response = val;
                keypoints.push_back(kp);
            }
        }
    }
    
    // Draw result - match Harris styling exactly
    ImageData result = img;
    for (const auto& kp : keypoints) {
        draw_circle(result, static_cast<int>(kp.x), static_cast<int>(kp.y), 3, 199, 30, 100);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    ShiTomasiResult hr;
    hr.keypoints = keypoints;
    hr.result_image = result;
    hr.time_ms = time_ms;
    hr.num_corners = keypoints.size();
    
    return hr;
}

// Helper function for elementwise product
static std::vector<std::vector<float>> compute_elementwise_product(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
    int h = a.size(), w = a[0].size();
    std::vector<std::vector<float>> result(h, std::vector<float>(w));
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            result[i][j] = a[i][j] * b[i][j];
        }
    }
    return result;
}
