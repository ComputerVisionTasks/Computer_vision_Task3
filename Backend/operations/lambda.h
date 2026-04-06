#ifndef LAMBDA_H
#define LAMBDA_H

#include "utils.h"
#include <vector>

// Same structure as HarrisResult, redefined for clarity.
// We could also reuse HarrisResult, but having a dedicated struct is cleaner.
struct ShiTomasiResult {
    std::vector<Keypoint> keypoints;
    ImageData result_image;
    double time_ms;
    int num_corners;
};

// Signature intentionally matches detect_harris_corners to allow easy swapping.
// The parameter 'k' is kept in the signature but ignored in Shi-Tomasi calc.
ShiTomasiResult detect_shi_tomasi(const ImageData& img, float k = 0.05f, int threshold = 1000000, int nms_size = 3);

#endif
