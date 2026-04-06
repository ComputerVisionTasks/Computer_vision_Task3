#include "matcher.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>

// ─────────────────────────────────────────────────────────────
//  Descriptor normalization
// ─────────────────────────────────────────────────────────────

static std::vector<float> normalize_descriptor(const std::vector<float>& d) {
    float mean = 0;
    for (float v : d) mean += v;
    mean /= static_cast<float>(d.size());

    float norm = 0;
    for (float v : d) norm += (v - mean) * (v - mean);
    norm = std::sqrt(norm);

    std::vector<float> out(d.size());
    if (norm < 1e-8f) {
        std::fill(out.begin(), out.end(), 0.0f);
    } else {
        for (size_t i = 0; i < d.size(); i++)
            out[i] = (d[i] - mean) / norm;
    }
    return out;
}

// ─────────────────────────────────────────────────────────────
//  Core metrics
// ─────────────────────────────────────────────────────────────

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
    return std::max(-1.0f, std::min(1.0f, dot));
}

// ─────────────────────────────────────────────────────────────
//  Color mapping
//
//  Maps a normalised quality value t ∈ [0, 1] to an RGB colour
//  using an HSV hue sweep:
//
//    t = 0  (best match)  →  hue 120° = green
//    t = 0.5              →  hue  60° = yellow
//    t = 1  (worst match) →  hue   0° = red
//
//  Saturation and value are fixed at 1 for maximum visibility
//  against any image background.
// ─────────────────────────────────────────────────────────────

static void quality_color(float t, uint8_t& r, uint8_t& g, uint8_t& b) {
    // Clamp t
    t = std::max(0.0f, std::min(1.0f, t));

    // Hue: 120° (green) → 0° (red) as t goes 0 → 1
    float hue = (1.0f - t) * 120.0f;   // degrees
    float s   = 1.0f;
    float v   = 1.0f;

    // HSV → RGB conversion
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

// ─────────────────────────────────────────────────────────────
//  Shared visualisation builder
//
//  Lines are drawn in order from best (lowest distance) to worst
//  (highest distance), which is also the sort order of `matches`,
//  so index position maps directly to the quality gradient.
//
//  Each line's color is computed from its rank within the drawn
//  set: rank 0 → green, rank (n-1) → red.
// ─────────────────────────────────────────────────────────────

static ImageData build_vis(const ImageData& img1, const ImageData& img2,
                           const std::vector<Match>& matches,
                           const std::vector<Keypoint>& kp1,
                           const std::vector<Keypoint>& kp2) {
    int total_width = img1.width + img2.width;
    int max_height  = std::max(img1.height, img2.height);
    ImageData vis   = create_blank_image(total_width, max_height, img1.channels);

    // Copy left image
    for (int y = 0; y < img1.height; y++)
        for (int x = 0; x < img1.width; x++) {
            int src = (y * img1.width  + x) * img1.channels;
            int dst = (y * total_width + x) * img1.channels;
            for (int c = 0; c < img1.channels; c++)
                vis.data[dst + c] = img1.data[src + c];
        }

    // Copy right image
    for (int y = 0; y < img2.height; y++)
        for (int x = 0; x < img2.width; x++) {
            int src = (y * img2.width  + x)              * img2.channels;
            int dst = (y * total_width + img1.width + x) * img1.channels;
            for (int c = 0; c < std::min(img1.channels, img2.channels); c++)
                vis.data[dst + c] = img2.data[src + c];
        }

    // Draw top-50 matches with quality-based colors
    int num_lines = std::min(50, static_cast<int>(matches.size()));
    for (int i = 0; i < num_lines; i++) {
        const auto& m = matches[i];

        // t = 0 for the best match, t = 1 for the worst drawn match
        float t = (num_lines > 1)
                  ? static_cast<float>(i) / static_cast<float>(num_lines - 1)
                  : 0.0f;

        uint8_t r, g, b;
        quality_color(t, r, g, b);

        draw_line(vis,
                  static_cast<int>(kp1[m.idx1].x), static_cast<int>(kp1[m.idx1].y),
                  static_cast<int>(kp2[m.idx2].x) + img1.width,
                  static_cast<int>(kp2[m.idx2].y),
                  r, g, b, 2);
    }

    return vis;
}

// ─────────────────────────────────────────────────────────────
//  SSD matching
// ─────────────────────────────────────────────────────────────

MatchingResult match_ssd(const std::vector<Keypoint>& kp1, const std::vector<Keypoint>& kp2,
                         const ImageData& img1, const ImageData& img2, float ratio_thresh) {
    auto start = std::chrono::high_resolution_clock::now();

    ratio_thresh = std::min(ratio_thresh, 0.75f);
    const float ABS_SSD_THRESH = 1.20f;

    std::vector<std::vector<float>> nd1(kp1.size()), nd2(kp2.size());
    for (size_t i = 0; i < kp1.size(); i++) nd1[i] = normalize_descriptor(kp1[i].descriptor);
    for (size_t j = 0; j < kp2.size(); j++) nd2[j] = normalize_descriptor(kp2[j].descriptor);

    struct Candidate { int idx; float best, second; };
    const float INF = std::numeric_limits<float>::max();
    std::vector<Candidate> fwd(kp1.size(), { -1, INF, INF });

    for (size_t i = 0; i < kp1.size(); i++)
        for (size_t j = 0; j < kp2.size(); j++) {
            float d = compute_ssd(nd1[i], nd2[j]);
            if (d < fwd[i].best) {
                fwd[i].second = fwd[i].best; fwd[i].best = d;
                fwd[i].idx    = static_cast<int>(j);
            } else if (d < fwd[i].second) {
                fwd[i].second = d;
            }
        }

    std::vector<int> rev(kp2.size(), -1);
    for (size_t j = 0; j < kp2.size(); j++) {
        float best = INF;
        for (size_t i = 0; i < kp1.size(); i++) {
            float d = compute_ssd(nd2[j], nd1[i]);
            if (d < best) { best = d; rev[j] = static_cast<int>(i); }
        }
    }

    std::vector<Match> matches;
    for (size_t i = 0; i < kp1.size(); i++) {
        int j = fwd[i].idx;
        if (j == -1)                                               continue;
        if (fwd[i].best > ABS_SSD_THRESH)                         continue;
        if (fwd[i].second < INF &&
            fwd[i].best / fwd[i].second >= ratio_thresh)          continue;
        if (rev[j] != static_cast<int>(i))                        continue;
        matches.push_back({ static_cast<int>(i), j, fwd[i].best });
    }

    std::sort(matches.begin(), matches.end(),
              [](const Match& a, const Match& b){ return a.distance < b.distance; });

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    MatchingResult mr;
    mr.matches       = matches;
    mr.visualization = build_vis(img1, img2, matches, kp1, kp2);
    mr.time_ms       = time_ms;
    mr.num_matches   = static_cast<int>(matches.size());
    return mr;
}

// ─────────────────────────────────────────────────────────────
//  NCC matching
// ─────────────────────────────────────────────────────────────

MatchingResult match_ncc(const std::vector<Keypoint>& kp1, const std::vector<Keypoint>& kp2,
                         const ImageData& img1, const ImageData& img2, float ratio_thresh) {
    auto start = std::chrono::high_resolution_clock::now();

    ratio_thresh = std::min(ratio_thresh, 0.80f);
    const float MIN_NCC = 0.65f;

    std::vector<std::vector<float>> nd1(kp1.size()), nd2(kp2.size());
    for (size_t i = 0; i < kp1.size(); i++) nd1[i] = normalize_descriptor(kp1[i].descriptor);
    for (size_t j = 0; j < kp2.size(); j++) nd2[j] = normalize_descriptor(kp2[j].descriptor);

    struct NccCandidate { int idx; float best, second; };
    std::vector<NccCandidate> fwd(kp1.size(), { -1, -2.0f, -2.0f });

    for (size_t i = 0; i < kp1.size(); i++)
        for (size_t j = 0; j < kp2.size(); j++) {
            float ncc = compute_ncc(nd1[i], nd2[j]);
            if (ncc > fwd[i].best) {
                fwd[i].second = fwd[i].best; fwd[i].best = ncc;
                fwd[i].idx    = static_cast<int>(j);
            } else if (ncc > fwd[i].second) {
                fwd[i].second = ncc;
            }
        }

    std::vector<int> rev(kp2.size(), -1);
    for (size_t j = 0; j < kp2.size(); j++) {
        float best = -2.0f;
        for (size_t i = 0; i < kp1.size(); i++) {
            float ncc = compute_ncc(nd2[j], nd1[i]);
            if (ncc > best) { best = ncc; rev[j] = static_cast<int>(i); }
        }
    }

    std::vector<Match> matches;
    for (size_t i = 0; i < kp1.size(); i++) {
        int j = fwd[i].idx;
        if (j == -1)                              continue;
        if (fwd[i].best < MIN_NCC)                continue;
        if (fwd[i].second > -2.0f) {
            float best_dist = 1.0f - fwd[i].best;
            float sec_dist  = 1.0f - fwd[i].second;
            if (sec_dist < 1e-6f)                 continue;
            if (best_dist / sec_dist >= ratio_thresh) continue;
        }
        if (rev[j] != static_cast<int>(i))        continue;
        matches.push_back({ static_cast<int>(i), j, 1.0f - fwd[i].best });
    }

    std::sort(matches.begin(), matches.end(),
              [](const Match& a, const Match& b){ return a.distance < b.distance; });

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    MatchingResult mr;
    mr.matches       = matches;
    mr.visualization = build_vis(img1, img2, matches, kp1, kp2);
    mr.time_ms       = time_ms;
    mr.num_matches   = static_cast<int>(matches.size());
    return mr;
}