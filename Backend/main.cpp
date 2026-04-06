#include "include/httplib.h"
#include "include/json.hpp"
#include "operations/harris.h"
#include "operations/sift.h"
#include "operations/matcher.h"
#include "operations/utils.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>

using json = nlohmann::json;
namespace fs = std::filesystem;

std::vector<ImageData> uploaded_images;
std::vector<HarrisResult> harris_results;
std::vector<SIFTResult> sift_results;
MatchingResult ssd_result;
MatchingResult ncc_result;

std::string get_base64_from_file(const std::string& path) {
    ImageData img = load_image(path);
    if (img.data.empty()) return "";
    return save_image_base64(img);
}

void load_case_study_images() {
    std::string case_study_path = "case_study/";
    if (fs::exists(case_study_path)) {
        for (const auto& entry : fs::directory_iterator(case_study_path)) {
            if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
                ImageData img = load_image(entry.path().string());
                if (!img.data.empty()) {
                    uploaded_images.push_back(resize_image(img, 640, 480));
                }
            }
        }
    }
    
    // If no images found, create dummy test images
    if (uploaded_images.empty()) {
        int w = 640, h = 480;
        ImageData dummy1 = create_blank_image(w, h, 3);
        ImageData dummy2 = create_blank_image(w, h, 3);
        
        // Draw some patterns
        for (int i = 0; i < 10; i++) {
            int x = 50 + i * 60;
            int y = 200;
            for (int dx = -10; dx <= 10; dx++) {
                for (int dy = -10; dy <= 10; dy++) {
                    if (dx*dx + dy*dy <= 100) {
                        int px = x + dx, py = y + dy;
                        if (px >= 0 && px < w && py >= 0 && py < h) {
                            int idx = (py * w + px) * 3;
                            dummy1.data[idx] = 255;
                            dummy1.data[idx+1] = 255;
                            dummy1.data[idx+2] = 255;
                        }
                    }
                }
            }
        }
        
        for (int i = 0; i < 10; i++) {
            int x = 100 + i * 55;
            int y = 250;
            for (int dx = -10; dx <= 10; dx++) {
                for (int dy = -10; dy <= 10; dy++) {
                    if (dx*dx + dy*dy <= 100) {
                        int px = x + dx, py = y + dy;
                        if (px >= 0 && px < w && py >= 0 && py < h) {
                            int idx = (py * w + px) * 3;
                            dummy2.data[idx] = 255;
                            dummy2.data[idx+1] = 255;
                            dummy2.data[idx+2] = 255;
                        }
                    }
                }
            }
        }
        
        uploaded_images.push_back(dummy1);
        uploaded_images.push_back(dummy2);
    }
}

int main() {
    httplib::Server svr;
    
    // Load case study images
    load_case_study_images();
    
    std::cout << "Server starting on port 8080..." << std::endl;
    std::cout << "Loaded " << uploaded_images.size() << " images" << std::endl;
    
    // Enable CORS
    svr.Options(".*", [](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.status = 204;
    });
    
    // Upload endpoint
    svr.Post("/api/upload", [](const httplib::Request& req, httplib::Response& res) {
        auto json_body = json::parse(req.body);
        
        if (json_body.contains("images") && json_body["images"].is_array()) {
            uploaded_images.clear();
            for (const auto& img_base64 : json_body["images"]) {
                std::string base64_str = img_base64.get<std::string>();
                ImageData img = decode_base64_image(base64_str);
                if (!img.data.empty()) {
                    uploaded_images.push_back(resize_image(img, 640, 480));
                }
            }
        }
        
        json response;
        response["success"] = true;
        response["count"] = uploaded_images.size();
        
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(response.dump(), "application/json");
    });
    
    // Harris endpoint
    svr.Post("/api/harris", [](const httplib::Request& req, httplib::Response& res) {
        json response;
        
        if (uploaded_images.size() < 2) {
            response["success"] = false;
            response["error"] = "Need at least 2 images";
            res.set_content(response.dump(), "application/json");
            return;
        }
        
        harris_results.clear();
        std::vector<json> harris_data;
        
        for (size_t i = 0; i < std::min(size_t(2), uploaded_images.size()); i++) {
            HarrisResult hr = detect_harris_corners(uploaded_images[i]);
            harris_results.push_back(hr);
            
            json result_json;
            result_json["image"] = save_image_base64(hr.result_image);
            result_json["num_corners"] = hr.num_corners;
            result_json["time_ms"] = hr.time_ms;
            harris_data.push_back(result_json);
        }
        
        response["success"] = true;
        response["results"] = harris_data;
        
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(response.dump(), "application/json");
    });
    
    // SIFT endpoint
    svr.Post("/api/sift", [](const httplib::Request& req, httplib::Response& res) {
        json response;
        
        if (uploaded_images.size() < 2) {
            response["success"] = false;
            response["error"] = "Need at least 2 images";
            res.set_content(response.dump(), "application/json");
            return;
        }
        
        sift_results.clear();
        std::vector<json> sift_data;
        
        for (size_t i = 0; i < std::min(size_t(2), uploaded_images.size()); i++) {
            SIFTResult sr = extract_sift_features(uploaded_images[i]);
            sift_results.push_back(sr);
            
            json result_json;
            result_json["image"] = save_image_base64(sr.result_image);
            result_json["num_keypoints"] = sr.num_keypoints;
            result_json["time_ms"] = sr.time_ms;
            sift_data.push_back(result_json);
        }
        
        response["success"] = true;
        response["results"] = sift_data;
        
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(response.dump(), "application/json");
    });
    
    // SSD Matching endpoint
    svr.Post("/api/match-ssd", [](const httplib::Request& req, httplib::Response& res) {
        json response;
        
        if (sift_results.size() < 2) {
            response["success"] = false;
            response["error"] = "Run SIFT first on at least 2 images";
            res.set_content(response.dump(), "application/json");
            return;
        }
        
        ssd_result = match_ssd(sift_results[0].keypoints, sift_results[1].keypoints,
                               uploaded_images[0], uploaded_images[1]);
        
        response["success"] = true;
        response["image"] = save_image_base64(ssd_result.visualization);
        response["num_matches"] = ssd_result.num_matches;
        response["time_ms"] = ssd_result.time_ms;
        
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(response.dump(), "application/json");
    });
    
    // NCC Matching endpoint
    svr.Post("/api/match-ncc", [](const httplib::Request& req, httplib::Response& res) {
        json response;
        
        if (sift_results.size() < 2) {
            response["success"] = false;
            response["error"] = "Run SIFT first on at least 2 images";
            res.set_content(response.dump(), "application/json");
            return;
        }
        
        ncc_result = match_ncc(sift_results[0].keypoints, sift_results[1].keypoints,
                               uploaded_images[0], uploaded_images[1]);
        
        response["success"] = true;
        response["image"] = save_image_base64(ncc_result.visualization);
        response["num_matches"] = ncc_result.num_matches;
        response["time_ms"] = ncc_result.time_ms;
        
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(response.dump(), "application/json");
    });
    
    svr.listen("0.0.0.0", 8080);
    
    return 0;
}