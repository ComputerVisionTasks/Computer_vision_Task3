#include "matcher.h"
#include <cmath>
#include <algorithm>
#include <limits>

float compute_ssd(const std::vector<float>& desc1, const std::vector<float>& desc2) {
    float sum = 0;
    for (size_t i = 0; i < desc1.size(); i++) {
        float diff = desc1[i] - desc2[i];
        sum += diff * diff;
    }
    return sum;
}

float compute_ncc(const std::vector<float>& desc1, const std::vector<float>& desc2) {
    float dot = 0, norm1 = 0, norm2 = 0;
    for (size_t i = 0; i < desc1.size(); i++) {
        dot += desc1[i] * desc2[i];
        norm1 += desc1[i] * desc1[i];
        norm2 += desc2[i] * desc2[i];
    }
    if (norm1 == 0 || norm2 == 0) return 0;
    return dot / (sqrt(norm1) * sqrt(norm2));
}

MatchingResult match_ssd(const std::vector<Keypoint>& kp1, const std::vector<Keypoint>& kp2,
                         const ImageData& img1, const ImageData& img2, float ratio_thresh) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<Match> matches;
    
    for (size_t i = 0; i < kp1.size(); i++) {
        float best_dist = std::numeric_limits<float>::max();
        float second_best = std::numeric_limits<float>::max();
        int best_idx = -1;
        
        for (size_t j = 0; j < kp2.size(); j++) {
            float dist = compute_ssd(kp1[i].descriptor, kp2[j].descriptor);
            if (dist < best_dist) {
                second_best = best_dist;
                best_dist = dist;
                best_idx = j;
            } else if (dist < second_best) {
                second_best = dist;
            }
        }
        
        if (best_idx != -1 && best_dist / second_best < ratio_thresh) {
            Match m;
            m.idx1 = i;
            m.idx2 = best_idx;
            m.distance = best_dist;
            matches.push_back(m);
        }
    }
    
    // Sort by distance
    std::sort(matches.begin(), matches.end(), [](const Match& a, const Match& b) {
        return a.distance < b.distance;
    });
    
    // Create side-by-side visualization
    int total_width = img1.width + img2.width;
    int max_height = std::max(img1.height, img2.height);
    ImageData vis = create_blank_image(total_width, max_height, img1.channels);
    
    // Copy images
    for (int y = 0; y < img1.height; y++) {
        for (int x = 0; x < img1.width; x++) {
            int src_idx = (y * img1.width + x) * img1.channels;
            int dst_idx = (y * total_width + x) * img1.channels;
            for (int c = 0; c < img1.channels; c++) {
                vis.data[dst_idx + c] = img1.data[src_idx + c];
            }
        }
    }
    
    for (int y = 0; y < img2.height; y++) {
        for (int x = 0; x < img2.width; x++) {
            int src_idx = (y * img2.width + x) * img2.channels;
            int dst_idx = (y * total_width + img1.width + x) * img2.channels;
            for (int c = 0; c < img2.channels; c++) {
                vis.data[dst_idx + c] = img2.data[src_idx + c];
            }
        }
    }
    
    // Draw matching lines (top 50 matches)
    int num_lines = std::min(50, static_cast<int>(matches.size()));
    for (int i = 0; i < num_lines; i++) {
        const auto& match = matches[i];
        int x1 = static_cast<int>(kp1[match.idx1].x);
        int y1 = static_cast<int>(kp1[match.idx1].y);
        int x2 = static_cast<int>(kp2[match.idx2].x) + img1.width;
        int y2 = static_cast<int>(kp2[match.idx2].y);
        
        // Color based on distance (better matches = greener)
        uint8_t r = static_cast<uint8_t>(std::min(255, static_cast<int>(match.distance * 50)));
        uint8_t g = static_cast<uint8_t>(std::min(255, static_cast<int>(255 - match.distance * 30)));
        uint8_t b = 100;
        
        draw_line(vis, x1, y1, x2, y2, r, g, b);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    MatchingResult mr;
    mr.matches = matches;
    mr.visualization = vis;
    mr.time_ms = time_ms;
    mr.num_matches = matches.size();
    
    return mr;
}

MatchingResult match_ncc(const std::vector<Keypoint>& kp1, const std::vector<Keypoint>& kp2,
                         const ImageData& img1, const ImageData& img2, float ratio_thresh) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<Match> matches;
    
    for (size_t i = 0; i < kp1.size(); i++) {
        float best_ncc = -1;
        float second_best = -1;
        int best_idx = -1;
        
        for (size_t j = 0; j < kp2.size(); j++) {
            float ncc = compute_ncc(kp1[i].descriptor, kp2[j].descriptor);
            if (ncc > best_ncc) {
                second_best = best_ncc;
                best_ncc = ncc;
                best_idx = j;
            } else if (ncc > second_best) {
                second_best = ncc;
            }
        }
        
        if (best_idx != -1 && best_ncc > ratio_thresh && second_best > 0) {
            if (best_ncc / second_best > 1.2f) {
                Match m;
                m.idx1 = i;
                m.idx2 = best_idx;
                m.distance = 1.0f - best_ncc; // Store as distance for sorting
                matches.push_back(m);
            }
        }
    }
    
    // Sort by distance (lower is better)
    std::sort(matches.begin(), matches.end(), [](const Match& a, const Match& b) {
        return a.distance < b.distance;
    });
    
    // Create side-by-side visualization (similar to SSD)
    int total_width = img1.width + img2.width;
    int max_height = std::max(img1.height, img2.height);
    ImageData vis = create_blank_image(total_width, max_height, img1.channels);
    
    // Copy images
    for (int y = 0; y < img1.height; y++) {
        for (int x = 0; x < img1.width; x++) {
            int src_idx = (y * img1.width + x) * img1.channels;
            int dst_idx = (y * total_width + x) * img1.channels;
            for (int c = 0; c < img1.channels; c++) {
                vis.data[dst_idx + c] = img1.data[src_idx + c];
            }
        }
    }
    
    for (int y = 0; y < img2.height; y++) {
        for (int x = 0; x < img2.width; x++) {
            int src_idx = (y * img2.width + x) * img2.channels;
            int dst_idx = (y * total_width + img1.width + x) * img2.channels;
            for (int c = 0; c < img2.channels; c++) {
                vis.data[dst_idx + c] = img2.data[src_idx + c];
            }
        }
    }
    
    // Draw matching lines
    int num_lines = std::min(50, static_cast<int>(matches.size()));
    for (int i = 0; i < num_lines; i++) {
        const auto& match = matches[i];
        int x1 = static_cast<int>(kp1[match.idx1].x);
        int y1 = static_cast<int>(kp1[match.idx1].y);
        int x2 = static_cast<int>(kp2[match.idx2].x) + img1.width;
        int y2 = static_cast<int>(kp2[match.idx2].y);
        
        // Cyan for NCC matches
        draw_line(vis, x1, y1, x2, y2, 0, 255, 255);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    MatchingResult mr;
    mr.matches = matches;
    mr.visualization = vis;
    mr.time_ms = time_ms;
    mr.num_matches = matches.size();
    
    return mr;
}