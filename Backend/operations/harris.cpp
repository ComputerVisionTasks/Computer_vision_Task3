#include "harris.h"
#include "utils.h"
#include <cmath>
#include <algorithm>
#include <queue>

#include <utility>


// ─────────────────────────────────────────────────────────────
//  Harris corner detector
//
//  Improvements over the previous version:
//    1. Proper 3×3 Sobel gradients (noise-resistant)
//    2. Adaptive threshold (% of max response)
//    3. Larger NMS window (7×7)
//    4. Top-500 cap on keypoints
//    5. Boundary-safe Gaussian blur
// ─────────────────────────────────────────────────────────────

HarrisResult detect_harris_corners(const ImageData& img, float k, int threshold, int nms_size) {
    auto start = std::chrono::high_resolution_clock::now();
    
    ImageData gray = img.channels > 1 ? grayscale(img) : img;
    auto [Ix, Iy] = compute_gradients(gray);
    
    // Compute structure tensor components with Gaussian blur
    auto Ixx = gaussian_blur(compute_elementwise_product(Ix, Ix), 1.5f);
    auto Iyy = gaussian_blur(compute_elementwise_product(Iy, Iy), 1.5f);
    auto Ixy = gaussian_blur(compute_elementwise_product(Ix, Iy), 1.5f);
    
    int w = gray.width, h = gray.height;
    
    // make a 2d vector with zeros
    std::vector<std::vector<float>> R(h, std::vector<float>(w, 0));
    
    // Compute Harris response R = det(M) - k * trace(M)^2
    float max_R = 0.0f;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            // det
            float det = Ixx[y][x] * Iyy[y][x] - Ixy[y][x] * Ixy[y][x];
            // trace
            float trace = Ixx[y][x] + Iyy[y][x];
            // Response 
            R[y][x] = det - k * trace * trace;
            if (R[y][x] > max_R) max_R = R[y][x];
        }
    }
    
    // Adaptive threshold: 1% of the maximum response value
    float adaptive_thresh = 0.01f * max_R;
    
    // Non-maximum suppression with larger window (7×7)
    nms_size = std::max(nms_size, 7);
    int nms_radius = nms_size / 2;
    
    std::vector<Keypoint> keypoints;
    
    for (int y = nms_radius; y < h - nms_radius; y++) {
        for (int x = nms_radius; x < w - nms_radius; x++) {
            float val = R[y][x];
            if (val < adaptive_thresh) continue;
            
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
    
    // Keep top 3000 by response strength
    std::sort(keypoints.begin(), keypoints.end(),
              [](const Keypoint& a, const Keypoint& b) {
                  return a.response > b.response;
              });
    if (keypoints.size() > 3000) keypoints.resize(3000);
    
    // Draw result
    ImageData result = img;
    for (const auto& kp : keypoints) {
        draw_circle(result, static_cast<int>(kp.x), static_cast<int>(kp.y), 3, 199, 30, 100);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    HarrisResult hr;
    hr.keypoints = keypoints;
    hr.result_image = result;
    hr.time_ms = time_ms;
    hr.num_corners = keypoints.size();
    
    return hr;
}
