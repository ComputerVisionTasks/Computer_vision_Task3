#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

namespace ImageUtils {
    struct StoredImage {
        int id;
        std::string filename;
        std::string base64_data;
        cv::Mat original_mat;
        int width;
        int height;
    };

    static std::vector<StoredImage> g_stored_images;
    static int g_next_id = 0;

    inline void addImage(const std::string& filename, const std::string& base64_data, const cv::Mat& img) {
        StoredImage stored;
        stored.id = g_next_id++;
        stored.filename = filename;
        stored.base64_data = base64_data;
        stored.original_mat = img.clone();
        stored.width = img.cols;
        stored.height = img.rows;
        g_stored_images.push_back(stored);
    }

    inline void clearImages() {
        g_stored_images.clear();
        g_next_id = 0;
    }

    inline const std::vector<StoredImage>& getAllImages() {
        return g_stored_images;
    }
}
