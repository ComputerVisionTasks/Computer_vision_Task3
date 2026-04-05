#ifndef HARRIS_H
#define HARRIS_H

#include "utils.h"
#include <vector>

struct HarrisResult {
    std::vector<Keypoint> keypoints;
    ImageData result_image;
    double time_ms;
    int num_corners;
};

HarrisResult detect_harris_corners(const ImageData& img, float k = 0.05f, int threshold = 1000000, int nms_size = 3);

#endif