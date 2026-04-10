
#include "sift.h"
#include "utils.h"
#include <cmath>
#include <algorithm>
#include <chrono>
#include <array>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ================================================================
//  IMAGE UTILITIES
// ================================================================

static std::vector<std::vector<float>> to_float(const ImageData& img) {
    ImageData gray = img.channels > 1 ? grayscale(img) : img;
    std::vector<std::vector<float>> out(gray.height, std::vector<float>(gray.width));
    // Convert [0,255] bytes to [0,1] float intensities for numeric processing.
    for (int y = 0; y < gray.height; y++)
        for (int x = 0; x < gray.width; x++)
            out[y][x] = gray.data[y * gray.width + x] / 255.0f;
    return out;
}

static void draw_hollow_circle_sift(
        ImageData& img,
        int center_x,
        int center_y,
        int radius,
        int thickness,
        uint8_t color_r,
        uint8_t color_g,
        uint8_t color_b) {

    if (radius <= 0) return;
    thickness = std::max(1, thickness);

    // Keep only the annulus [inner_radius, outer_radius] to draw a hollow ring.
    int outer_radius_sq = radius * radius;
    int inner_radius = std::max(0, radius - thickness);
    int inner_radius_sq = inner_radius * inner_radius;

    for (int offset_y = -radius; offset_y <= radius; offset_y++) {
        for (int offset_x = -radius; offset_x <= radius; offset_x++) {
            int squared_distance = offset_x * offset_x + offset_y * offset_y;
            if (squared_distance <= outer_radius_sq && squared_distance >= inner_radius_sq) {
                int pixel_x = center_x + offset_x;
                int pixel_y = center_y + offset_y;
                if (pixel_x >= 0 && pixel_x < img.width && pixel_y >= 0 && pixel_y < img.height) {
                    int pixel_index = (pixel_y * img.width + pixel_x) * img.channels;
                    if (img.channels >= 3) {
                        img.data[pixel_index] = color_r;
                        img.data[pixel_index + 1] = color_g;
                        img.data[pixel_index + 2] = color_b;
                    }
                }
            }
        }
    }
}

// ================================================================
//  DOWNSAMPLE (anti-aliased)
// ================================================================

static std::vector<std::vector<float>> downsample(
        const std::vector<std::vector<float>>& img) {

    // pre-blur to avoid aliasing, then take every other pixel
    auto blurred = gaussian_blur(img, 1.0f);
    int h = (int)blurred.size() / 2;
    int w = (int)blurred[0].size() / 2;
    std::vector<std::vector<float>> out(h, std::vector<float>(w));
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            out[y][x] = blurred[y * 2][x * 2];
    return out;
}

// ================================================================
//  DIFFERENCE OF GAUSSIANS
// ================================================================

static std::vector<std::vector<float>> dog(
        const std::vector<std::vector<float>>& a,
        const std::vector<std::vector<float>>& b) {

    int h = (int)a.size(), w = (int)a[0].size();
    std::vector<std::vector<float>> out(h, std::vector<float>(w));
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            out[y][x] = b[y][x] - a[y][x];
    return out;
}

// ================================================================
//  SUBPIXEL REFINEMENT — proper 3×3 linear solve
//  Solves: H * offset = -grad  where H is the 3×3 Hessian
//  Returns false if the point should be discarded.
// ================================================================

static bool refine(
        const std::vector<std::vector<std::vector<float>>>& D,
        int l, int y, int x,
        float& xr, float& yr, float& lr, float& interp_val) {

    int h = (int)D[l].size(), w = (int)D[l][0].size();
    if (x < 1 || x >= w-1 || y < 1 || y >= h-1) return false;
    if (l < 1 || l >= (int)D.size()-1)            return false;

    // ---- gradient ----
    float dx = (D[l][y][x+1] - D[l][y][x-1]) * 0.5f;
    float dy = (D[l][y+1][x] - D[l][y-1][x]) * 0.5f;
    float ds = (D[l+1][y][x] - D[l-1][y][x]) * 0.5f;

    // ---- Hessian ----
    float v   =  D[l  ][y  ][x  ];
    float dxx =  D[l  ][y  ][x+1] - 2.0f*v + D[l  ][y  ][x-1];
    float dyy =  D[l  ][y+1][x  ] - 2.0f*v + D[l  ][y-1][x  ];
    float dss =  D[l+1][y  ][x  ] - 2.0f*v + D[l-1][y  ][x  ];
    float dxy = (D[l  ][y+1][x+1] - D[l  ][y+1][x-1]
               - D[l  ][y-1][x+1] + D[l  ][y-1][x-1]) * 0.25f;
    float dxs = (D[l+1][y  ][x+1] - D[l+1][y  ][x-1]
               - D[l-1][y  ][x+1] + D[l-1][y  ][x-1]) * 0.25f;
    float dys = (D[l+1][y+1][x  ] - D[l+1][y-1][x  ]
               - D[l-1][y+1][x  ] + D[l-1][y-1][x  ]) * 0.25f;

    // ---- solve 3×3 system via Cramer's rule ----
    // H = | dxx dxy dxs |     grad = | dx |
    //     | dxy dyy dys |            | dy |
    //     | dxs dys dss |            | ds |
    float det = dxx*(dyy*dss - dys*dys)
              - dxy*(dxy*dss - dys*dxs)
              + dxs*(dxy*dys - dyy*dxs);

    if (std::fabs(det) < 1e-10f) return false;

    float inv = 1.0f / det;
    // cofactor matrix (symmetric)
    float A00 =  (dyy*dss - dys*dys) * inv;
    float A01 = -(dxy*dss - dys*dxs) * inv;
    float A02 =  (dxy*dys - dyy*dxs) * inv;
    float A11 =  (dxx*dss - dxs*dxs) * inv;
    float A12 = -(dxx*dys - dxs*dxy) * inv;
    float A22 =  (dxx*dyy - dxy*dxy) * inv;

    xr = -(A00*dx + A01*dy + A02*ds);
    yr = -(A01*dx + A11*dy + A12*ds);
    lr = -(A02*dx + A12*dy + A22*ds);

    if (std::fabs(xr) > 0.5f || std::fabs(yr) > 0.5f || std::fabs(lr) > 0.5f)
        return false;

    // interpolated response
    interp_val = v + 0.5f*(dx*xr + dy*yr + ds*lr);
    return true;
}

// ================================================================
//  EDGE RESPONSE FILTER
//  Reject if ratio of principal curvatures > threshold (r=10)
// ================================================================

static bool edge_response_ok(
        const std::vector<std::vector<float>>& dog_layer,
        int y, int x, float r_thresh = 7.0f) {

    float v   = dog_layer[y][x];
    float dxx = dog_layer[y][x+1] - 2.0f*v + dog_layer[y][x-1];
    float dyy = dog_layer[y+1][x] - 2.0f*v + dog_layer[y-1][x];
    float dxy = (dog_layer[y+1][x+1] - dog_layer[y+1][x-1]
               - dog_layer[y-1][x+1] + dog_layer[y-1][x-1]) * 0.25f;

    float trace = dxx + dyy;
    float det   = dxx * dyy - dxy * dxy;

    if (det <= 0.0f) return false;  // negative curvature

    float ratio = (trace * trace) / det;
    float thresh = (r_thresh + 1.0f) * (r_thresh + 1.0f) / r_thresh;
    return ratio < thresh;
}

// ================================================================
//  ORIENTATION ASSIGNMENT
//  Smoothed 36-bin histogram; returns dominant orientations.
// ================================================================

static std::vector<float> orientations(
        const std::vector<std::vector<float>>& img,
        float cx, float cy, float scale) {

    const int bins  = 36;
    std::vector<float> hist(bins, 0.0f);

    int   r     = (int)std::ceil(3.0f * 1.5f * scale);
    float sigma = 1.5f * scale;

    int h = (int)img.size(), w = (int)img[0].size();

    for (int dy = -r; dy <= r; dy++)
    for (int dx = -r; dx <= r; dx++) {
        int px = (int)std::round(cx) + dx;
        int py = (int)std::round(cy) + dy;
        if (px <= 0 || py <= 0 || px >= w-1 || py >= h-1) continue;

        float gx  = img[py][px+1] - img[py][px-1];
        float gy  = img[py+1][px] - img[py-1][px];
        float mag = std::sqrt(gx*gx + gy*gy);
        float ang = std::atan2(gy, gx);  // [-π, π]

        float w2  = std::exp(-(float)(dx*dx + dy*dy) / (2.0f * sigma * sigma));

        // map angle to [0, 36)
        int b = (int)std::floor((ang + (float)M_PI) / (2.0f*(float)M_PI) * bins);
        b = std::clamp(b, 0, bins-1);
        hist[b] += mag * w2;
    }

    // ---- smooth histogram (3×, circular) ----
    for (int iter = 0; iter < 3; iter++) {
        std::vector<float> tmp(bins);
        for (int i = 0; i < bins; i++)
            tmp[i] = (hist[(i-1+bins)%bins] + hist[i]*2.0f + hist[(i+1)%bins]) * 0.25f;
        hist = tmp;
    }

    float maxv = *std::max_element(hist.begin(), hist.end());
    if (maxv < 1e-7f) return {};

    std::vector<float> out;
    for (int i = 0; i < bins; i++) {
        float prev = hist[(i-1+bins)%bins];
        float curr = hist[i];
        float next = hist[(i+1)%bins];

        // local peak?
        if (curr < 0.8f * maxv)      continue;
        if (curr < prev || curr < next) continue;

        // parabolic peak interpolation
        float offset = 0.5f * (prev - next) / (prev - 2.0f*curr + next + 1e-8f);
        float angle  = 2.0f*(float)M_PI * ((float)i + offset) / (float)bins - (float)M_PI;
        out.push_back(angle);
    }
    return out;
}

// ================================================================
//  DESCRIPTOR — 4×4 spatial grid, 8 orientation bins
//  Gaussian-weighted, trilinear interpolation across cells & bins
// ================================================================

static std::vector<float> descriptor(
        const std::vector<std::vector<float>>& img,
        float cx, float cy, float scale, float ori) {

    const int  grid = 4, obins = 8;
    const float cell = 4.0f * scale;   // pixels per spatial cell
    const float sigma_desc = 0.5f * (float)grid * cell;

    std::vector<float> desc(grid * grid * obins, 0.0f);

    float ct = std::cos(-ori), st = std::sin(-ori);
    int   r  = (int)std::ceil(cell * (float)grid * 0.5f * std::sqrt(2.0f));

    int h = (int)img.size(), w = (int)img[0].size();

    for (int dy = -r; dy <= r; dy++)
    for (int dx = -r; dx <= r; dx++) {
        int px = (int)std::round(cx) + dx;
        int py = (int)std::round(cy) + dy;
        if (px <= 0 || py <= 0 || px >= w-1 || py >= h-1) continue;

        // rotate into keypoint frame
        float rx = ( ct*(float)dx + st*(float)dy) / cell + (float)grid*0.5f - 0.5f;
        float ry = (-st*(float)dx + ct*(float)dy) / cell + (float)grid*0.5f - 0.5f;
        if (rx < -1.0f || ry < -1.0f || rx >= (float)grid || ry >= (float)grid) continue;

        float gx  = img[py][px+1] - img[py][px-1];
        float gy  = img[py+1][px] - img[py-1][px];
        float mag  = std::sqrt(gx*gx + gy*gy);
        float ang  = std::atan2(gy, gx) - ori;

        while (ang < -(float)M_PI) ang += 2.0f*(float)M_PI;
        while (ang >  (float)M_PI) ang -= 2.0f*(float)M_PI;

        // Gaussian spatial weight
        float dist2 = (float)(dx*dx + dy*dy) / (sigma_desc * sigma_desc);
        float wgt   = std::exp(-0.5f * dist2) * mag;

        // normalised orientation in [0, obins)
        float ob = (ang + (float)M_PI) / (2.0f*(float)M_PI) * (float)obins;

        // ---- trilinear interpolation ----
        // spatial weights
        float fx = rx - std::floor(rx), fy = ry - std::floor(ry);
        int   ix = (int)std::floor(rx), iy = (int)std::floor(ry);

        // orientation weights
        float fo  = ob - std::floor(ob);
        int   ib  = ((int)std::floor(ob) % obins + obins) % obins;
        int   ib1 = (ib + 1) % obins;

        // distribute into up to 8 cells
        for (int ci = 0; ci <= 1; ci++)
        for (int cj = 0; cj <= 1; cj++) {
            int xi = ix + ci, yi = iy + cj;
            if (xi < 0 || xi >= grid || yi < 0 || yi >= grid) continue;

            float sw = wgt
                     * (ci == 0 ? 1.0f - fx : fx)
                     * (cj == 0 ? 1.0f - fy : fy);

            int base = (yi * grid + xi) * obins;
            desc[base + ib ] += sw * (1.0f - fo);
            desc[base + ib1] += sw * fo;
        }
    }

    // ---- L2 normalise, clamp at 0.2, re-normalise ----
    auto l2norm = [](const std::vector<float>& v) {
        float s = 0;
        for (float x : v) s += x*x;
        return std::sqrt(s) + 1e-7f;
    };

    float n = l2norm(desc);
    for (float& v : desc) v /= n;
    for (float& v : desc) v  = std::min(v, 0.2f);
    n = l2norm(desc);
    for (float& v : desc) v /= n;

    return desc;
}

// ================================================================
//  MAIN ENTRY POINT
// ================================================================

SIFTResult extract_sift_features(const ImageData& img) {

    // Start timing (for performance measurement)
    auto start_time = std::chrono::high_resolution_clock::now();

    // Convert image to float for precise computations
    auto base_image = to_float(img);

    // ---- SIFT parameters ----
    const int octave_count = 4;          // Number of pyramid layers (multi-resolution)
    const int levels_per_octave = 6;     // Number of blur levels per octave
    const float base_sigma = 1.6f;       // Initial Gaussian blur (standard in SIFT)

    // k = scale factor between levels
    const float scale_step = std::pow(2.0f, 1.0f / (float)(levels_per_octave - 3));

    // ---- build scale-space ----
    // gaussian_pyramid[octave][level] = blurred image
    // dog_pyramid[octave][level]      = Difference of Gaussian images
    using Img2D  = std::vector<std::vector<float>>;
    using ImgSet = std::vector<Img2D>;

    std::vector<ImgSet> gaussian_pyramid, dog_pyramid;

    // First image is blurred with base sigma
    Img2D octave_seed = gaussian_blur(base_image, base_sigma);

    // Loop over octaves (multi-scale representation)
    for (int octave_index = 0; octave_index < octave_count; octave_index++) {

        ImgSet gaussian_levels;

        // First level of octave
        gaussian_levels.push_back(octave_seed);

        float previous_sigma = base_sigma;

        // Build Gaussian pyramid levels
        for (int level_index = 1; level_index < levels_per_octave; level_index++) {

            // Desired sigma for this level
            float target_sigma = base_sigma * std::pow(scale_step, (float)level_index);

            // Incremental blur (difference from previous level)
            float incremental_sigma = std::sqrt(std::max(1e-4f,
                target_sigma * target_sigma - previous_sigma * previous_sigma));

            // Apply blur incrementally
            gaussian_levels.push_back(
                gaussian_blur(gaussian_levels.back(), incremental_sigma)
            );

            previous_sigma = target_sigma;
        }

        // Store Gaussian levels
        gaussian_pyramid.push_back(gaussian_levels);

        // ---- Build DoG pyramid ----
        ImgSet dog_levels;

        for (int level_index = 1; level_index < levels_per_octave; level_index++) {
            // DoG = difference between consecutive Gaussian images
            dog_levels.push_back(
                dog(gaussian_levels[level_index - 1], gaussian_levels[level_index])
            );
        }

        dog_pyramid.push_back(dog_levels);

        // ---- Prepare next octave ----
        // Take a middle level and downsample (reduce resolution)
        if (octave_index < octave_count - 1)
            octave_seed = downsample(gaussian_levels[levels_per_octave - 3]);
    }

    // ---- detect & describe keypoints ----
    std::vector<Keypoint> keypoints;

    // Threshold to remove weak points (low contrast)
    const float contrast_threshold = 0.06f / (float)(levels_per_octave - 3);

    // Loop over octaves
    for (int octave_index = 0; octave_index < octave_count; octave_index++) {

        int dog_level_count = (int)dog_pyramid[octave_index].size();
        int image_height = (int)dog_pyramid[octave_index][0].size();
        int image_width  = (int)dog_pyramid[octave_index][0][0].size();

        // Skip first and last level (need neighbors in scale)
        for (int level_index = 1; level_index < dog_level_count - 1; level_index++)
        for (int y = 1; y < image_height - 1; y++)
        for (int x = 1; x < image_width - 1; x++) {

            float dog_value = dog_pyramid[octave_index][level_index][y][x];

            // ---- Step 1: quick contrast filtering ----
            if (std::fabs(dog_value) < 0.5f * contrast_threshold) continue;

            // ---- Step 2: check if pixel is local extrema ----
            // Compare with 26 neighbors (3x3x3 cube in scale-space)
            bool is_extremum = true;

            for (int d_level = -1; d_level <= 1 && is_extremum; d_level++)
            for (int d_row   = -1; d_row <= 1 && is_extremum; d_row++)
            for (int d_col   = -1; d_col <= 1; d_col++) {

                // Skip center point
                if (!d_level && !d_row && !d_col) continue;

                float neighbor_value =
                    dog_pyramid[octave_index][level_index + d_level][y + d_row][x + d_col];

                // Check max or min condition
                if ((dog_value > 0 && neighbor_value >= dog_value) ||
                    (dog_value < 0 && neighbor_value <= dog_value)) {
                    is_extremum = false;
                }
            }

            if (!is_extremum) continue;

            // ---- Step 3: refine keypoint location ----
            float x_offset, y_offset, level_offset, interpolated_value;

            if (!refine(dog_pyramid[octave_index], level_index, y, x,
                        x_offset, y_offset, level_offset, interpolated_value)) {
                continue; // discard unstable points
            }

            // Reject low contrast after refinement
            if (std::fabs(interpolated_value) < contrast_threshold) continue;

            // ---- Step 4: remove edge-like points ----
            if (!edge_response_ok(dog_pyramid[octave_index][level_index], y, x))
                continue;

            // ---- Step 5: compute scale of keypoint ----
            float keypoint_scale =
                base_sigma *
                std::pow(scale_step, (float)level_index + level_offset) *
                std::pow(2.0f, (float)octave_index);

            // Get corresponding Gaussian image
            const Img2D& gaussian_image = gaussian_pyramid[octave_index][level_index];

            // Refined position
            float refined_x = (float)x + x_offset;
            float refined_y = (float)y + y_offset;

            // ---- Step 6: assign orientation(s) ----
            auto dominant_orientations = orientations(
                gaussian_image,
                refined_x,
                refined_y,
                base_sigma * std::pow(scale_step, (float)level_index)
            );

            // ---- Step 7: build descriptor ----
            for (float orientation : dominant_orientations) {

                Keypoint keypoint;

                // Convert coordinates to original image scale
                keypoint.x = refined_x * std::pow(2.0f, (float)octave_index);
                keypoint.y = refined_y * std::pow(2.0f, (float)octave_index);

                keypoint.scale = keypoint_scale;
                keypoint.response = interpolated_value;
                keypoint.orientation = orientation;

                // Descriptor = 128-dim vector
                keypoint.descriptor = descriptor(
                    gaussian_image,
                    refined_x,
                    refined_y,
                    base_sigma * std::pow(scale_step, (float)level_index),
                    orientation
                );

                keypoints.push_back(keypoint);
            }
        }
    }

    // keep top 500 by |response|
    std::sort(keypoints.begin(), keypoints.end(),
              [](const Keypoint& a, const Keypoint& b) {
                  return std::fabs(a.response) > std::fabs(b.response);
              });
    if (keypoints.size() > 500) keypoints.resize(500);

    // ---- visualise ----
    ImageData output_image = img;
    for (const auto& keypoint : keypoints) {
        // Convert detected scale to a display radius while preserving the old bold minimum size.
        int blob_radius = std::max(4, (int)std::lround(keypoint.scale));
        draw_hollow_circle_sift(
            output_image,
            (int)keypoint.x,
            (int)keypoint.y,
            blob_radius,
            2,
            199,
            30,
            100);

        // Small orientation arrow for direction visualization.
        int arrow_end_x = (int)(keypoint.x + 8.0f * std::cos(keypoint.orientation));
        int arrow_end_y = (int)(keypoint.y + 8.0f * std::sin(keypoint.orientation));
        draw_line(output_image, (int)keypoint.x, (int)keypoint.y, arrow_end_x, arrow_end_y, 199, 30, 100);
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    SIFTResult result;
    result.keypoints = keypoints;
    result.result_image = output_image;
    result.time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    result.num_keypoints = (int)keypoints.size();
    return result;
}