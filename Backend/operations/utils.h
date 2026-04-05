#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdint>

struct ImageData {
    int width;
    int height;
    int channels;
    std::vector<uint8_t> data;
};

struct Point2D {
    float x, y;
    Point2D() : x(0), y(0) {}
    Point2D(float _x, float _y) : x(_x), y(_y) {}
};

struct Keypoint {
    float x, y;
    float scale;
    float orientation;
    std::vector<float> descriptor;
    float response;
    
    Keypoint() : x(0), y(0), scale(1.0f), orientation(0.0f), response(0.0f) {}
};

ImageData load_image(const std::string& path);
std::string save_image_base64(const ImageData& img);
ImageData create_blank_image(int width, int height, int channels);
ImageData grayscale(const ImageData& img);
ImageData resize_image(const ImageData& img, int new_width, int new_height);
std::string base64_encode(const std::vector<uint8_t>& data);
std::vector<uint8_t> base64_decode(const std::string& encoded);
void draw_circle(ImageData& img, int x, int y, int radius, uint8_t r, uint8_t g, uint8_t b);
void draw_line(ImageData& img, int x1, int y1, int x2, int y2, uint8_t r, uint8_t g, uint8_t b);
ImageData decode_base64_image(const std::string& base64_str);

#endif