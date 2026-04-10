#include "utils.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"
#include <sstream>
#include <iomanip>
#include <cstring>

ImageData load_image(const std::string& path) {
    ImageData img;
    int width, height, channels;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 0);
    if (data) {
        img.width = width;
        img.height = height;
        img.channels = channels;
        img.data.assign(data, data + (width * height * channels));
        stbi_image_free(data);
    }
    return img;
}

ImageData grayscale(const ImageData& img) {
    ImageData gray;
    gray.width = img.width;
    gray.height = img.height;
    gray.channels = 1;
    gray.data.resize(img.width * img.height);
    
    for (int i = 0; i < img.height; i++) {
        for (int j = 0; j < img.width; j++) {
            int idx = (i * img.width + j) * img.channels;
            uint8_t r = img.data[idx];
            uint8_t g = img.data[idx + 1];
            uint8_t b = img.data[idx + 2];
            gray.data[i * img.width + j] = static_cast<uint8_t>(0.299f * r + 0.587f * g + 0.114f * b);
        }
    }
    return gray;
}

std::string save_image_base64(const ImageData& img) {
    std::vector<uint8_t> png_data;
    stbi_write_png_to_func([](void* context, void* data, int size) {
        std::vector<uint8_t>* vec = static_cast<std::vector<uint8_t>*>(context);
        vec->insert(vec->end(), static_cast<uint8_t*>(data), static_cast<uint8_t*>(data) + size);
    }, &png_data, img.width, img.height, img.channels, img.data.data(), 0);
    
    return base64_encode(png_data);
}

std::string base64_encode(const std::vector<uint8_t>& data) {
    static const char* chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    int i = 0;
    uint32_t octet_a, octet_b, octet_c;
    
    for (size_t j = 0; j < data.size(); j += 3) {
        octet_a = j < data.size() ? data[j] : 0;
        octet_b = j + 1 < data.size() ? data[j + 1] : 0;
        octet_c = j + 2 < data.size() ? data[j + 2] : 0;
        
        uint32_t triple = (octet_a << 16) + (octet_b << 8) + octet_c;
        
        result += chars[(triple >> 18) & 0x3F];
        result += chars[(triple >> 12) & 0x3F];
        result += chars[(triple >> 6) & 0x3F];
        result += chars[triple & 0x3F];
    }
    
    int mod = data.size() % 3;
    if (mod == 1) {
        result[result.size() - 1] = '=';
        result[result.size() - 2] = '=';
    } else if (mod == 2) {
        result[result.size() - 1] = '=';
    }
    
    return result;
}

void draw_circle(ImageData& img, int cx, int cy, int radius, uint8_t r, uint8_t g, uint8_t b) {
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            if (x*x + y*y <= radius*radius) {
                int px = cx + x;
                int py = cy + y;
                if (px >= 0 && px < img.width && py >= 0 && py < img.height) {
                    int idx = (py * img.width + px) * img.channels;
                    if (img.channels >= 3) {
                        img.data[idx] = r;
                        img.data[idx + 1] = g;
                        img.data[idx + 2] = b;
                    }
                }
            }
        }
    }
}

void draw_line(ImageData& img, int x1, int y1, int x2, int y2, uint8_t r, uint8_t g, uint8_t b, int thickness) {
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int sx = x1 < x2 ? 1 : -1;
    int sy = y1 < y2 ? 1 : -1;
    int err = dx - dy;
    
    int x = x1, y = y1;
    while (true) {
        // Draw a square of size (thickness x thickness) centered at (x, y)
        int offset = thickness / 2;
        for (int ty = -offset; ty <= offset; ty++) {
            for (int tx = -offset; tx <= offset; tx++) {
                int px = x + tx;
                int py = y + ty;
                if (px >= 0 && px < img.width && py >= 0 && py < img.height) {
                    int idx = (py * img.width + px) * img.channels;
                    if (img.channels >= 3) {
                        img.data[idx] = r;
                        img.data[idx + 1] = g;
                        img.data[idx + 2] = b;
                    }
                }
            }
        }
        
        if (x == x2 && y == y2) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x += sx; }
        if (e2 < dx) { err += dx; y += sy; }
    }
}

ImageData decode_base64_image(const std::string& base64_str) {
    std::vector<uint8_t> decoded = base64_decode(base64_str);
    int width, height, channels;
    unsigned char* data = stbi_load_from_memory(decoded.data(), decoded.size(), &width, &height, &channels, 0);
    ImageData img;
    if (data) {
        img.width = width;
        img.height = height;
        img.channels = channels;
        img.data.assign(data, data + (width * height * channels));
        stbi_image_free(data);
    }
    return img;
}

std::vector<uint8_t> base64_decode(const std::string& encoded) {
    static const std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<uint8_t> result;
    int i = 0;
    uint32_t buffer = 0;
    int bits = 0;
    
    for (char c : encoded) {
        if (c == '=') break;
        size_t idx = chars.find(c);
        if (idx == std::string::npos) continue;
        
        buffer = (buffer << 6) | idx;
        bits += 6;
        
        if (bits >= 8) {
            bits -= 8;
            result.push_back((buffer >> bits) & 0xFF);
        }
    }
    return result;
}

ImageData create_blank_image(int width, int height, int channels) {
    ImageData img;
    img.width = width;
    img.height = height;
    img.channels = channels;
    img.data.resize(width * height * channels, channels == 1 ? 128 : 255);
    return img;
}

// ─────────────────────────────────────────────────────────────
//  Shared Gaussian utilities
// ─────────────────────────────────────────────────────────────

std::vector<float> gaussian_kernel(float sigma) {
    int r    = (int)std::ceil(3.0f * sigma);
    int size = 2 * r + 1;
    std::vector<float> k(size);
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float x = (float)(i - r);
        k[i] = std::exp(-(x * x) / (2.0f * sigma * sigma));
        sum += k[i];
    }
    for (auto& v : k) v /= sum;
    return k;
}

std::vector<std::vector<float>> gaussian_blur(
        const std::vector<std::vector<float>>& img, float sigma) {

    int h = (int)img.size(), w = (int)img[0].size();
    auto k = gaussian_kernel(sigma);
    int r  = (int)k.size() / 2;

    // horizontal pass
    std::vector<std::vector<float>> tmp(h, std::vector<float>(w, 0.0f));
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            for (int i = -r; i <= r; i++) {
                int xx = std::clamp(x + i, 0, w - 1);
                tmp[y][x] += img[y][xx] * k[i + r];
            }

    // vertical pass
    std::vector<std::vector<float>> out(h, std::vector<float>(w, 0.0f));
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            for (int i = -r; i <= r; i++) {
                int yy = std::clamp(y + i, 0, h - 1);
                out[y][x] += tmp[yy][x] * k[i + r];
            }
    return out;
}

std::vector<std::vector<float>> compute_elementwise_product(
        const std::vector<std::vector<float>>& a,
        const std::vector<std::vector<float>>& b) {
    int h = (int)a.size(), w = (int)a[0].size();
    std::vector<std::vector<float>> result(h, std::vector<float>(w));
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
            result[i][j] = a[i][j] * b[i][j];
    return result;
}

ImageData resize_image(const ImageData& img, int new_width, int new_height) {
    if (img.width == new_width && img.height == new_height) return img;
    ImageData resized;
    resized.width = new_width;
    resized.height = new_height;
    resized.channels = img.channels;
    resized.data.resize(new_width * new_height * img.channels);

    float scale_x = static_cast<float>(img.width - 1) / (new_width - 1 > 0 ? new_width - 1 : 1);
    float scale_y = static_cast<float>(img.height - 1) / (new_height - 1 > 0 ? new_height - 1 : 1);

    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            float gx = x * scale_x;
            float gy = y * scale_y;
            int gxi = static_cast<int>(gx);
            int gyi = static_cast<int>(gy);
            int gxi1 = std::min(gxi + 1, img.width - 1);
            int gyi1 = std::min(gyi + 1, img.height - 1);
            float dx = gx - gxi;
            float dy = gy - gyi;

            for (int c = 0; c < img.channels; c++) {
                float c00 = img.data[(gyi * img.width + gxi) * img.channels + c];
                float c10 = img.data[(gyi * img.width + gxi1) * img.channels + c];
                float c01 = img.data[(gyi1 * img.width + gxi) * img.channels + c];
                float c11 = img.data[(gyi1 * img.width + gxi1) * img.channels + c];

                float val = c00 * (1 - dx) * (1 - dy) +
                            c10 * dx * (1 - dy) +
                            c01 * (1 - dx) * dy +
                            c11 * dx * dy;
                resized.data[(y * new_width + x) * img.channels + c] = static_cast<uint8_t>(val);
            }
        }
    }
    return resized;
}