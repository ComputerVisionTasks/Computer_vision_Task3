#include "sift.h"
#include <cmath>
#include <algorithm>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Gaussian pyramid level
struct OctaveLevel {
    std::vector<std::vector<float>> image;
    float sigma;
};

std::vector<std::vector<float>> image_to_float(const ImageData& img) {
    ImageData gray = img.channels > 1 ? grayscale(img) : img;
    std::vector<std::vector<float>> result(gray.height, std::vector<float>(gray.width));
    for (int y = 0; y < gray.height; y++) {
        for (int x = 0; x < gray.width; x++) {
            result[y][x] = static_cast<float>(gray.data[y * gray.width + x]) / 255.0f;
        }
    }
    return result;
}

ImageData float_to_image(const std::vector<std::vector<float>>& img) {
    ImageData result;
    result.height = img.size();
    result.width = img[0].size();
    result.channels = 1;
    result.data.resize(result.width * result.height);
    for (int y = 0; y < result.height; y++) {
        for (int x = 0; x < result.width; x++) {
            result.data[y * result.width + x] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, img[y][x] * 255.0f)));
        }
    }
    return result;
}

std::vector<std::vector<float>> gaussian_blur_float(const std::vector<std::vector<float>>& img, float sigma) {
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

std::vector<std::vector<float>> compute_dog(const std::vector<std::vector<float>>& g1, const std::vector<std::vector<float>>& g2) {
    int h = g1.size(), w = g1[0].size();
    std::vector<std::vector<float>> result(h, std::vector<float>(w));
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            result[y][x] = g2[y][x] - g1[y][x];
        }
    }
    return result;
}

std::vector<Keypoint> detect_extrema(const std::vector<std::vector<std::vector<std::vector<float>>>>& dog_pyramid, float threshold) {
    std::vector<Keypoint> keypoints;
    
    for (size_t octave = 0; octave < dog_pyramid.size(); octave++) {
        int levels = dog_pyramid[octave].size();
        int h = dog_pyramid[octave][0].size();
        int w = dog_pyramid[octave][0][0].size();
        
        for (int level = 1; level < levels - 1; level++) {
            for (int y = 1; y < h - 1; y++) {
                for (int x = 1; x < w - 1; x++) {
                    float val = dog_pyramid[octave][level][y][x];
                    if (fabs(val) < threshold) continue;
                    
                    bool is_extremum = true;
                    // Check 26 neighbors
                    for (int dl = -1; dl <= 1 && is_extremum; dl++) {
                        for (int dy = -1; dy <= 1 && is_extremum; dy++) {
                            for (int dx = -1; dx <= 1; dx++) {
                                if (dl == 0 && dy == 0 && dx == 0) continue;
                                int nl = level + dl;
                                if (nl >= 0 && nl < levels) {
                                    int ny = y + dy, nx = x + dx;
                                    if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
                                        if ((val > 0 && dog_pyramid[octave][nl][ny][nx] > val) ||
                                            (val < 0 && dog_pyramid[octave][nl][ny][nx] < val)) {
                                            is_extremum = false;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    if (is_extremum) {
                        Keypoint kp;
                        kp.x = static_cast<float>(x) * powf(2.0f, octave);
                        kp.y = static_cast<float>(y) * powf(2.0f, octave);
                        kp.scale = 1.6f * powf(2.0f, octave);
                        kp.response = val;
                        keypoints.push_back(kp);
                    }
                }
            }
        }
    }
    
    return keypoints;
}

std::vector<float> compute_descriptor(const std::vector<std::vector<float>>& img, float x, float y, float scale) {
    const int num_bins = 8;
    const int num_histograms = 4;
    std::vector<float> descriptor(num_histograms * num_histograms * num_bins, 0.0f);
    
    float sigma = 0.5f * scale;
    int radius = static_cast<int>(3 * sigma);
    float sigma_ori = 1.5f * scale;
    
    // Compute gradients in the region
    std::vector<std::vector<float>> histograms(num_histograms, std::vector<float>(num_histograms * num_bins, 0.0f));
    
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int px = static_cast<int>(x + dx);
            int py = static_cast<int>(y + dy);
            if (px < 1 || px >= img[0].size() - 1 || py < 1 || py >= img.size() - 1) continue;
            
            // Compute gradient
            float grad_x = img[py][px + 1] - img[py][px - 1];
            float grad_y = img[py + 1][px] - img[py - 1][px];
            float magnitude = sqrt(grad_x * grad_x + grad_y * grad_y);
            float orientation = atan2(grad_y, grad_x);
            
            // Gaussian weight
            float dist = sqrt(dx * dx + dy * dy);
            float weight = exp(-(dist * dist) / (2 * sigma_ori * sigma_ori));
            
            // Bin orientation
            int bin = static_cast<int>((orientation + M_PI) / (2 * M_PI) * num_bins) % num_bins;
            
            // Histogram bin index
            int hx = static_cast<int>((dx + radius) * num_histograms / (2.0f * radius));
            int hy = static_cast<int>((dy + radius) * num_histograms / (2.0f * radius));
            hx = std::min(std::max(hx, 0), num_histograms - 1);
            hy = std::min(std::max(hy, 0), num_histograms - 1);
            
            histograms[hy][hx * num_bins + bin] += magnitude * weight;
        }
    }
    
    // Flatten and normalize
    int idx = 0;
    for (int i = 0; i < num_histograms; i++) {
        for (int j = 0; j < num_histograms; j++) {
            for (int k = 0; k < num_bins; k++) {
                descriptor[idx++] = histograms[i][j * num_bins + k];
            }
        }
    }
    
    // Normalize
    float norm = 0;
    for (float v : descriptor) norm += v * v;
    norm = sqrt(norm);
    if (norm > 0) {
        for (float& v : descriptor) v /= norm;
    }
    
    // Clip and renormalize
    for (float& v : descriptor) v = std::min(v, 0.2f);
    norm = 0;
    for (float v : descriptor) norm += v * v;
    norm = sqrt(norm);
    if (norm > 0) {
        for (float& v : descriptor) v /= norm;
    }
    
    return descriptor;
}

SIFTResult extract_sift_features(const ImageData& img) {
    auto start = std::chrono::high_resolution_clock::now();
    
    auto float_img = image_to_float(img);
    
    // Build Gaussian pyramid
    const int num_octaves = 4;
    const int num_levels = 5;
    const float sigma0 = 1.6f;
    const float k = powf(2.0f, 1.0f / (num_levels - 1));
    
    std::vector<std::vector<std::vector<std::vector<float>>>> gaussian_pyramid;
    std::vector<std::vector<std::vector<std::vector<float>>>> dog_pyramid;
    
    std::vector<std::vector<float>> current = gaussian_blur_float(float_img, sigma0);
    
    for (int octave = 0; octave < num_octaves; octave++) {
        std::vector<std::vector<std::vector<float>>> gaussian_levels;
        gaussian_levels.push_back(current);
        
        for (int level = 1; level < num_levels; level++) {
            float sigma = sigma0 * powf(k, level);
            current = gaussian_blur_float(current, sigma);
            gaussian_levels.push_back(current);
        }
        
        gaussian_pyramid.push_back(gaussian_levels);
        
        // Compute DoG
        std::vector<std::vector<std::vector<float>>> dog_levels;
        for (int level = 1; level < num_levels; level++) {
            dog_levels.push_back(compute_dog(gaussian_levels[level - 1], gaussian_levels[level]));
        }
        dog_pyramid.push_back(dog_levels);
        
        // Downsample for next octave
        if (octave < num_octaves - 1) {
            int new_h = gaussian_levels[0].size() / 2;
            int new_w = gaussian_levels[0][0].size() / 2;
            std::vector<std::vector<float>> downsampled(new_h, std::vector<float>(new_w));
            for (int y = 0; y < new_h; y++) {
                for (int x = 0; x < new_w; x++) {
                    downsampled[y][x] = gaussian_levels[0][y * 2][x * 2];
                }
            }
            current = downsampled;
        }
    }
    
    // Detect extrema
    auto keypoints = detect_extrema(dog_pyramid, 0.03f);
    
    // Compute descriptors
    for (auto& kp : keypoints) {
        int octave = static_cast<int>(log2(kp.scale / 1.6f));
        float scale_in_octave = kp.scale / powf(2.0f, octave);
        
        std::vector<std::vector<float>> img_at_scale;
        if (octave < static_cast<int>(gaussian_pyramid.size())) {
            img_at_scale = gaussian_pyramid[octave][std::min(2, static_cast<int>(gaussian_pyramid[octave].size() - 1))];
        } else {
            img_at_scale = float_img;
        }
        
        float x_in_octave = kp.x / powf(2.0f, octave);
        float y_in_octave = kp.y / powf(2.0f, octave);
        
        kp.descriptor = compute_descriptor(img_at_scale, x_in_octave, y_in_octave, scale_in_octave);
    }
    
    // Sort by response and keep top 500
    std::sort(keypoints.begin(), keypoints.end(), [](const Keypoint& a, const Keypoint& b) {
        return fabs(a.response) > fabs(b.response);
    });
    if (keypoints.size() > 500) keypoints.resize(500);
    
    // Draw result
    ImageData result = img;
    for (const auto& kp : keypoints) {
        draw_circle(result, static_cast<int>(kp.x), static_cast<int>(kp.y), 4, 255, 0, 255);
        // Draw orientation line
        int x2 = static_cast<int>(kp.x + 8 * cos(kp.orientation));
        int y2 = static_cast<int>(kp.y + 8 * sin(kp.orientation));
        draw_line(result, static_cast<int>(kp.x), static_cast<int>(kp.y), x2, y2, 255, 255, 0);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    SIFTResult sr;
    sr.keypoints = keypoints;
    sr.result_image = result;
    sr.time_ms = time_ms;
    sr.num_keypoints = keypoints.size();
    
    return sr;
}