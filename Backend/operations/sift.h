#ifndef SIFT_H
#define SIFT_H

#include "utils.h"
#include <vector>

struct SIFTResult {
    std::vector<Keypoint> keypoints;
    ImageData result_image;
    double time_ms;
    int num_keypoints;
};

SIFTResult extract_sift_features(const ImageData& img);

#endif