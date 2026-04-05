#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "json.hpp"

using json = nlohmann::json;

namespace FeatureMatching {
    struct MatchResult {
        bool success;
        long long computation_time_ms;
        int num_matches;
        double avg_distance;
        std::string visualization_base64;
        json details;
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

    inline MatchResult matchSSD(const cv::Mat& desc1, const cv::Mat& desc2,
                               const std::vector<cv::KeyPoint>& kp1,
                               const std::vector<cv::KeyPoint>& kp2,
                               const cv::Mat& img1, const cv::Mat& img2) {
        MatchResult result;
        result.success = false;
        result.computation_time_ms = 0;
        result.num_matches = 0;
        result.avg_distance = 0.0;
        result.details = json::object();

        if (desc1.empty() || desc2.empty() || kp1.empty() || kp2.empty()) {
            result.details["error"] = "Invalid descriptors or keypoints";
            return result;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        // Brute force matching using SSD (Sum of Squared Differences)
        std::vector<cv::DMatch> matches;
        double total_distance = 0.0;

        for (int i = 0; i < desc1.rows; i++) {
            double min_distance = std::numeric_limits<double>::max();
            int best_match = -1;

            for (int j = 0; j < desc2.rows; j++) {
                double distance = 0.0;
                for (int k = 0; k < desc1.cols; k++) {
                    float diff = desc1.at<float>(i, k) - desc2.at<float>(j, k);
                    distance += diff * diff;
                }
                distance = std::sqrt(distance);

                if (distance < min_distance) {
                    min_distance = distance;
                    best_match = j;
                }
            }

            if (best_match >= 0 && min_distance < 500.0) {
                cv::DMatch match;
                match.queryIdx = i;
                match.trainIdx = best_match;
                match.distance = min_distance;
                matches.push_back(match);
                total_distance += min_distance;
            }
        }

        result.num_matches = matches.size();
        result.avg_distance = matches.empty() ? 0.0 : total_distance / matches.size();

        // Draw matches
        cv::Mat match_img;
        cv::drawMatches(img1, kp1, img2, kp2, matches, match_img);
        result.visualization_base64 = matToBase64(match_img);

        auto end_time = std::chrono::high_resolution_clock::now();
        result.computation_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        result.success = true;

        result.details["method"] = "SSD (Sum of Squared Differences)";
        result.details["keypoints_img1"] = (int)kp1.size();
        result.details["keypoints_img2"] = (int)kp2.size();
        result.details["matched_pairs"] = result.num_matches;

        return result;
    }

    inline MatchResult matchNCC(const cv::Mat& desc1, const cv::Mat& desc2,
                               const std::vector<cv::KeyPoint>& kp1,
                               const std::vector<cv::KeyPoint>& kp2,
                               const cv::Mat& img1, const cv::Mat& img2) {
        MatchResult result;
        result.success = false;
        result.computation_time_ms = 0;
        result.num_matches = 0;
        result.avg_distance = 0.0;
        result.details = json::object();

        if (desc1.empty() || desc2.empty() || kp1.empty() || kp2.empty()) {
            result.details["error"] = "Invalid descriptors or keypoints";
            return result;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        // Normalize descriptors for NCC
        cv::Mat desc1_norm, desc2_norm;
        cv::normalize(desc1, desc1_norm, 1, 0, cv::NORM_L2);
        cv::normalize(desc2, desc2_norm, 1, 0, cv::NORM_L2);

        // Brute force matching using NCC (Normalized Cross Correlation)
        std::vector<cv::DMatch> matches;
        double total_correlation = 0.0;

        for (int i = 0; i < desc1_norm.rows; i++) {
            double max_correlation = -1.0;
            int best_match = -1;

            for (int j = 0; j < desc2_norm.rows; j++) {
                double correlation = 0.0;
                for (int k = 0; k < desc1_norm.cols; k++) {
                    correlation += desc1_norm.at<float>(i, k) * desc2_norm.at<float>(j, k);
                }

                if (correlation > max_correlation) {
                    max_correlation = correlation;
                    best_match = j;
                }
            }

            if (best_match >= 0 && max_correlation > 0.7) {
                cv::DMatch match;
                match.queryIdx = i;
                match.trainIdx = best_match;
                match.distance = 1.0 - max_correlation; // Convert correlation to distance
                matches.push_back(match);
                total_correlation += max_correlation;
            }
        }

        result.num_matches = matches.size();
        result.avg_distance = matches.empty() ? 0.0 : total_correlation / matches.size();

        // Draw matches
        cv::Mat match_img;
        cv::drawMatches(img1, kp1, img2, kp2, matches, match_img);
        result.visualization_base64 = matToBase64(match_img);

        auto end_time = std::chrono::high_resolution_clock::now();
        result.computation_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        result.success = true;

        result.details["method"] = "NCC (Normalized Cross Correlation)";
        result.details["keypoints_img1"] = (int)kp1.size();
        result.details["keypoints_img2"] = (int)kp2.size();
        result.details["matched_pairs"] = result.num_matches;

        return result;
    }
}
