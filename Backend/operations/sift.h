#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "json.hpp"

using json = nlohmann::json;

namespace SIFTFeature {
    struct SIFTResult {
        bool success;
        long long computation_time_ms;
        int total_keypoints;
        std::vector<std::string> result_images_base64;
        json per_image_data;
    };

    inline std::string matToBase64(const cv::Mat& img) {
        std::vector<uchar> buf;
        cv::imencode(".png", img, buf);
        std::string base64_data = "data:image/png;base64,";
        static const char* base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        
        size_t i = 0;
        unsigned char char_array_3[3];
        unsigned char char_array_4[4];
        
        while (i < buf.size()) {
            char_array_3[0] = buf[i++];
            char_array_3[1] = (i < buf.size()) ? buf[i++] : 0;
            char_array_3[2] = (i < buf.size()) ? buf[i++] : 0;
            
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;
            
            for (int j = 0; j < 4; j++) {
                base64_data += (i <= buf.size() + j) ? base64_chars[char_array_4[j]] : '=';
            }
        }
        
        return base64_data;
    }

    inline SIFTResult extractAndDraw(const std::vector<cv::Mat>& images) {
        SIFTResult result;
        result.success = false;
        result.total_keypoints = 0;
        result.computation_time_ms = 0;
        result.per_image_data = json::array();
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        if (images.empty()) {
            return result;
        }

        cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

        for (const auto& img : images) {
            if (img.empty()) continue;

            cv::Mat gray;
            if (img.channels() == 3) {
                cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = img.clone();
            }

            // Detect keypoints and compute descriptors using SIFT
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            sift->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);

            // Draw keypoints on original image
            cv::Mat result_img;
            cv::drawKeypoints(img, keypoints, result_img, cv::Scalar(0, 255, 0),
                            cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

            result.total_keypoints += keypoints.size();
            result.result_images_base64.push_back(matToBase64(result_img));

            json img_data;
            img_data["keypoints_detected"] = (int)keypoints.size();
            img_data["image_width"] = img.cols;
            img_data["image_height"] = img.rows;
            if (!descriptors.empty()) {
                img_data["descriptor_size"] = descriptors.rows;
                img_data["descriptor_dims"] = descriptors.cols;
            }
            result.per_image_data.push_back(img_data);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        result.computation_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        result.success = true;

        return result;
    }
}
