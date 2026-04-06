#include "matcher.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>

// ═══════════════════════════════════════════════════════════════
//  DESCRIPTOR NORMALIZATION
//
//  Zero-mean, unit-variance normalization of feature descriptors.
//  This ensures that matching is invariant to global brightness
//  and contrast differences between the two images.
//
//  Steps:
//    1. Subtract the mean    → centres the descriptor at zero
//    2. Divide by the norm   → scales to unit length
//
//  If the descriptor has near-zero variance (flat patch), we
//  return an all-zeros vector to avoid division-by-zero.
// ═══════════════════════════════════════════════════════════════

static std::vector<float> normalize_descriptor(const std::vector<float>& d) {
    // Step 1: Compute the mean of all descriptor elements
    float mean = 0;
    for (float v : d) mean += v;
    mean /= static_cast<float>(d.size());

    // Step 2: Compute the L2 norm of the mean-subtracted descriptor
    float norm = 0;
    for (float v : d) norm += (v - mean) * (v - mean);
    norm = std::sqrt(norm);

    // Step 3: Produce the normalized descriptor (zero-mean, unit-norm)
    std::vector<float> out(d.size());
    if (norm < 1e-8f) {
        // Near-zero variance → featureless patch, return zeros
        std::fill(out.begin(), out.end(), 0.0f);
    } else {
        for (size_t i = 0; i < d.size(); i++)
            out[i] = (d[i] - mean) / norm;
    }
    return out;
}

// ═══════════════════════════════════════════════════════════════
//  DISTANCE / SIMILARITY METRICS
//
//  SSD (Sum of Squared Differences):
//    - Measures Euclidean distance² between two descriptors
//    - Lower = more similar.  SSD = 0 means identical.
//
//  NCC (Normalized Cross-Correlation):
//    - Dot product of two already-normalized descriptors
//    - Higher = more similar.  NCC = 1.0 means identical.
//    - Clamped to [-1, 1] for numerical safety.
// ═══════════════════════════════════════════════════════════════

static float compute_ssd(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

static float compute_ncc(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0;
    for (size_t i = 0; i < a.size(); i++) dot += a[i] * b[i];
    return std::max(-1.0f, std::min(1.0f, dot));  // Clamp to [-1, 1]
}

// ═══════════════════════════════════════════════════════════════
//  LINE COLOR GENERATION
//
//  Each match line needs a unique, highly distinguishable color.
//  We use the "golden angle" technique:
//
//    hue = (index × 137.508°) mod 360°
//
//  The golden angle (≈137.508°) is irrational with respect to 360°,
//  so consecutive indices produce hues that are maximally spread
//  around the color wheel — avoiding clusters of similar colors.
//
//  HSV is then converted to RGB with S=0.9, V=0.95 for vivid
//  but pleasant colors that stand out against any image background.
// ═══════════════════════════════════════════════════════════════

static void distinct_color(int idx, uint8_t& r, uint8_t& g, uint8_t& b) {
    float hue = std::fmod(idx * 137.508f, 360.0f);
    float s   = 0.9f;
    float v   = 0.95f;

    // Standard HSV → RGB conversion
    float h = hue / 60.0f;
    int   i = static_cast<int>(std::floor(h)) % 6;
    float f = h - std::floor(h);
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float k = v * (1.0f - s * (1.0f - f));

    float rf, gf, bf;
    switch (i) {
        case 0: rf = v; gf = k; bf = p; break;
        case 1: rf = q; gf = v; bf = p; break;
        case 2: rf = p; gf = v; bf = k; break;
        case 3: rf = p; gf = q; bf = v; break;
        case 4: rf = k; gf = p; bf = v; break;
        default:rf = v; gf = p; bf = q; break;
    }

    r = static_cast<uint8_t>(rf * 255);
    g = static_cast<uint8_t>(gf * 255);
    b = static_cast<uint8_t>(bf * 255);
}

// ═══════════════════════════════════════════════════════════════
//  MATCH VISUALIZATION BUILDER
//
//  Creates a side-by-side image of img1 (left) and img2 (right),
//  then draws the top-50 match lines connecting keypoints across
//  the two images.
//
//  For each match:
//    1. A colored line (thickness 3) connects the two keypoints
//    2. Filled circles (radius 8) mark both endpoints
//    3. Color is unique per line via the golden-angle palette
//
//  Matches are assumed to be sorted by quality (best first),
//  so the top-50 are the most confident correspondences.
// ═══════════════════════════════════════════════════════════════

static ImageData build_vis(const ImageData& img1, const ImageData& img2,
                           const std::vector<Match>& matches,
                           const std::vector<Keypoint>& kp1,
                           const std::vector<Keypoint>& kp2) {
    int total_width = img1.width + img2.width;
    int max_height  = std::max(img1.height, img2.height);
    ImageData vis   = create_blank_image(total_width, max_height, img1.channels);

    // Copy image 1 into the left half of the visualization canvas
    for (int y = 0; y < img1.height; y++)
        for (int x = 0; x < img1.width; x++) {
            int src = (y * img1.width  + x) * img1.channels;
            int dst = (y * total_width + x) * img1.channels;
            for (int c = 0; c < img1.channels; c++)
                vis.data[dst + c] = img1.data[src + c];
        }

    // Copy image 2 into the right half (offset by img1.width)
    // Uses min(channels) to handle mismatched channel counts safely
    for (int y = 0; y < img2.height; y++)
        for (int x = 0; x < img2.width; x++) {
            int src = (y * img2.width  + x)              * img2.channels;
            int dst = (y * total_width + img1.width + x) * img1.channels;
            for (int c = 0; c < std::min(img1.channels, img2.channels); c++)
                vis.data[dst + c] = img2.data[src + c];
        }

    // Draw the top-50 matches as colored lines with endpoint circles
    int num_lines = std::min(50, static_cast<int>(matches.size()));
    for (int i = 0; i < num_lines; i++) {
        const auto& m = matches[i];

        // Generate a unique color for this match line
        uint8_t r, g, b;
        distinct_color(i, r, g, b);

        // Keypoint coordinates (img2's x is offset by img1's width)
        int x1 = static_cast<int>(kp1[m.idx1].x);
        int y1 = static_cast<int>(kp1[m.idx1].y);
        int x2 = static_cast<int>(kp2[m.idx2].x) + img1.width;
        int y2 = static_cast<int>(kp2[m.idx2].y);

        // Draw the connecting line (thickness = 3 pixels)
        draw_line(vis, x1, y1, x2, y2, r, g, b, 3);

        // Draw filled circles at both endpoints (radius = 8 pixels)
        draw_circle(vis, x1, y1, 8, r, g, b);
        draw_circle(vis, x2, y2, 8, r, g, b);
    }

    return vis;
}

// ═══════════════════════════════════════════════════════════════
//  SSD MATCHING  (Sum of Squared Differences)
//
//  Algorithm:
//    1. Normalize all descriptors (zero-mean, unit-norm)
//    2. Forward pass: for each keypoint in kp1, find the best
//       and second-best match in kp2 by SSD distance
//    3. Reverse pass: for each keypoint in kp2, find its single
//       best match in kp1
//    4. Apply three filters to keep only reliable matches:
//       a) Absolute threshold — reject if SSD > 1.20
//       b) Ratio test (Lowe's) — reject if best/second_best >= 0.75
//          This ensures the best match is significantly better
//          than the runner-up, indicating a distinctive feature.
//       c) Cross-check — reject if kp2's best match for j is
//          not kp1[i].  Both directions must agree.
//    5. Sort surviving matches by distance (best first)
// ═══════════════════════════════════════════════════════════════

MatchingResult match_ssd(const std::vector<Keypoint>& kp1, const std::vector<Keypoint>& kp2,
                         const ImageData& img1, const ImageData& img2, float ratio_thresh) {
    auto start = std::chrono::high_resolution_clock::now();

    // Cap ratio threshold to prevent overly permissive matching
    ratio_thresh = std::min(ratio_thresh, 0.75f);

    // Matches with SSD above this value are too dissimilar to trust
    const float ABS_SSD_THRESH = 1.20f;

    // --- Step 1: Normalize all descriptors ---
    std::vector<std::vector<float>> nd1(kp1.size()), nd2(kp2.size());
    for (size_t i = 0; i < kp1.size(); i++) nd1[i] = normalize_descriptor(kp1[i].descriptor);
    for (size_t j = 0; j < kp2.size(); j++) nd2[j] = normalize_descriptor(kp2[j].descriptor);

    // --- Step 2: Forward pass (kp1 → kp2) ---
    // For each keypoint in image 1, find the closest and second-closest
    // match in image 2 by SSD distance
    struct Candidate { int idx; float best, second; };
    const float INF = std::numeric_limits<float>::max();
    std::vector<Candidate> fwd(kp1.size(), { -1, INF, INF });

    for (size_t i = 0; i < kp1.size(); i++)
        for (size_t j = 0; j < kp2.size(); j++) {
            float d = compute_ssd(nd1[i], nd2[j]);
            if (d < fwd[i].best) {
                fwd[i].second = fwd[i].best;  // Demote old best to second
                fwd[i].best   = d;
                fwd[i].idx    = static_cast<int>(j);
            } else if (d < fwd[i].second) {
                fwd[i].second = d;             // Update second-best only
            }
        }

    // --- Step 3: Reverse pass (kp2 → kp1) for cross-checking ---
    // For each keypoint in image 2, find its single best match in image 1
    std::vector<int> rev(kp2.size(), -1);
    for (size_t j = 0; j < kp2.size(); j++) {
        float best = INF;
        for (size_t i = 0; i < kp1.size(); i++) {
            float d = compute_ssd(nd2[j], nd1[i]);
            if (d < best) { best = d; rev[j] = static_cast<int>(i); }
        }
    }

    // --- Step 4: Apply filters and collect surviving matches ---
    std::vector<Match> matches;
    for (size_t i = 0; i < kp1.size(); i++) {
        int j = fwd[i].idx;

        // Filter (a): No valid match found
        if (j == -1)                                               continue;

        // Filter (b): Absolute distance too high — features are too different
        if (fwd[i].best > ABS_SSD_THRESH)                         continue;

        // Filter (c): Lowe's ratio test — best match must be significantly
        //             better than second-best to confirm distinctive feature
        if (fwd[i].second < INF &&
            fwd[i].best / fwd[i].second >= ratio_thresh)          continue;

        // Filter (d): Cross-check — kp2[j]'s best match must also be kp1[i]
        //             (bidirectional agreement prevents many-to-one matches)
        if (rev[j] != static_cast<int>(i))                        continue;

        matches.push_back({ static_cast<int>(i), j, fwd[i].best });
    }

    // --- Step 5: Sort by distance (most confident matches first) ---
    std::sort(matches.begin(), matches.end(),
              [](const Match& a, const Match& b){ return a.distance < b.distance; });

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Build the side-by-side visualization with match lines
    MatchingResult mr;
    mr.matches       = matches;
    mr.visualization = build_vis(img1, img2, matches, kp1, kp2);
    mr.time_ms       = time_ms;
    mr.num_matches   = static_cast<int>(matches.size());
    return mr;
}

// ═══════════════════════════════════════════════════════════════
//  NCC MATCHING  (Normalized Cross-Correlation)
//
//  Same overall structure as SSD, but uses NCC similarity instead:
//    - NCC is a similarity measure (higher = better), not a distance
//    - Best match has the HIGHEST NCC, not the lowest distance
//    - Ratio test uses (1 - NCC) as a pseudo-distance:
//        (1 - best_ncc) / (1 - second_best_ncc) >= threshold → reject
//
//  Filters applied:
//    a) Minimum NCC threshold (0.65) — reject weak correlations
//    b) Ratio test on converted distances
//    c) Cross-check (bidirectional agreement)
// ═══════════════════════════════════════════════════════════════

MatchingResult match_ncc(const std::vector<Keypoint>& kp1, const std::vector<Keypoint>& kp2,
                         const ImageData& img1, const ImageData& img2, float ratio_thresh) {
    auto start = std::chrono::high_resolution_clock::now();

    // Cap ratio threshold
    ratio_thresh = std::min(ratio_thresh, 0.80f);

    // Minimum NCC score to accept — below this the correlation is too weak
    const float MIN_NCC = 0.65f;

    // --- Step 1: Normalize all descriptors ---
    std::vector<std::vector<float>> nd1(kp1.size()), nd2(kp2.size());
    for (size_t i = 0; i < kp1.size(); i++) nd1[i] = normalize_descriptor(kp1[i].descriptor);
    for (size_t j = 0; j < kp2.size(); j++) nd2[j] = normalize_descriptor(kp2[j].descriptor);

    // --- Step 2: Forward pass (kp1 → kp2) ---
    // For each keypoint in image 1, find the highest and second-highest
    // NCC score among all keypoints in image 2
    struct NccCandidate { int idx; float best, second; };
    std::vector<NccCandidate> fwd(kp1.size(), { -1, -2.0f, -2.0f });

    for (size_t i = 0; i < kp1.size(); i++)
        for (size_t j = 0; j < kp2.size(); j++) {
            float ncc = compute_ncc(nd1[i], nd2[j]);
            if (ncc > fwd[i].best) {
                fwd[i].second = fwd[i].best;  // Demote old best to second
                fwd[i].best   = ncc;
                fwd[i].idx    = static_cast<int>(j);
            } else if (ncc > fwd[i].second) {
                fwd[i].second = ncc;           // Update second-best only
            }
        }

    // --- Step 3: Reverse pass (kp2 → kp1) for cross-checking ---
    std::vector<int> rev(kp2.size(), -1);
    for (size_t j = 0; j < kp2.size(); j++) {
        float best = -2.0f;
        for (size_t i = 0; i < kp1.size(); i++) {
            float ncc = compute_ncc(nd2[j], nd1[i]);
            if (ncc > best) { best = ncc; rev[j] = static_cast<int>(i); }
        }
    }

    // --- Step 4: Apply filters and collect surviving matches ---
    std::vector<Match> matches;
    for (size_t i = 0; i < kp1.size(); i++) {
        int j = fwd[i].idx;

        // Filter (a): No valid match found
        if (j == -1)                              continue;

        // Filter (b): NCC score too low — weak correlation
        if (fwd[i].best < MIN_NCC)                continue;

        // Filter (c): Ratio test — convert NCC to pseudo-distances
        //   best_dist  = 1 - best_ncc  (small if best match is strong)
        //   sec_dist   = 1 - sec_ncc   (larger if second match is weaker)
        //   Reject if best_dist / sec_dist is too close to 1.0
        //   (means best and second-best are similarly good → ambiguous)
        if (fwd[i].second > -2.0f) {
            float best_dist = 1.0f - fwd[i].best;
            float sec_dist  = 1.0f - fwd[i].second;
            if (sec_dist < 1e-6f)                 continue;  // Avoid division by zero
            if (best_dist / sec_dist >= ratio_thresh) continue;
        }

        // Filter (d): Cross-check — bidirectional agreement
        if (rev[j] != static_cast<int>(i))        continue;

        // Store as distance = (1 - NCC) so lower = better, consistent with SSD
        matches.push_back({ static_cast<int>(i), j, 1.0f - fwd[i].best });
    }

    // --- Step 5: Sort by distance (most confident matches first) ---
    std::sort(matches.begin(), matches.end(),
              [](const Match& a, const Match& b){ return a.distance < b.distance; });

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Build the side-by-side visualization with match lines
    MatchingResult mr;
    mr.matches       = matches;
    mr.visualization = build_vis(img1, img2, matches, kp1, kp2);
    mr.time_ms       = time_ms;
    mr.num_matches   = static_cast<int>(matches.size());
    return mr;
}