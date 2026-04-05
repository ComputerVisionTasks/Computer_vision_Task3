#ifndef MATCHER_H
#define MATCHER_H

#include "utils.h"
#include <vector>
#include <utility>

struct Match {
    int idx1, idx2;
    float distance;
};

struct MatchingResult {
    std::vector<Match> matches;
    ImageData visualization;
    double time_ms;
    int num_matches;
};

MatchingResult match_ssd(const std::vector<Keypoint>& kp1, const std::vector<Keypoint>& kp2, 
                         const ImageData& img1, const ImageData& img2, float ratio_thresh = 0.75f);

MatchingResult match_ncc(const std::vector<Keypoint>& kp1, const std::vector<Keypoint>& kp2,
                         const ImageData& img1, const ImageData& img2, float ratio_thresh = 0.75f);

#endif