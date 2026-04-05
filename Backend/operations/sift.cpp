
#include "sift.h"
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
    for (int y = 0; y < gray.height; y++)
        for (int x = 0; x < gray.width; x++)
            out[y][x] = gray.data[y * gray.width + x] / 255.0f;
    return out;
}

// ================================================================
//  GAUSSIAN BLUR
// ================================================================

static std::vector<float> gaussian_kernel(float sigma) {
    int r    = (int)std::ceil(3.0f * sigma);
    int size = 2 * r + 1;
    std::vector<float> k(size);
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float x = (float)(i - r);
        k[i] = std::exp(-(x * x) / (2.0f * sigma * sigma));
        sum += k[i];
    }
    for (auto& v : k) v /= sum;
    return k;
}

static std::vector<std::vector<float>> gaussian_blur(
        const std::vector<std::vector<float>>& img, float sigma) {

    int h = (int)img.size(), w = (int)img[0].size();
    auto k = gaussian_kernel(sigma);
    int r  = (int)k.size() / 2;

    // horizontal pass
    std::vector<std::vector<float>> tmp(h, std::vector<float>(w, 0.0f));
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            for (int i = -r; i <= r; i++) {
                int xx = std::clamp(x + i, 0, w - 1);
                tmp[y][x] += img[y][xx] * k[i + r];
            }

    // vertical pass
    std::vector<std::vector<float>> out(h, std::vector<float>(w, 0.0f));
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            for (int i = -r; i <= r; i++) {
                int yy = std::clamp(y + i, 0, h - 1);
                out[y][x] += tmp[yy][x] * k[i + r];
            }
    return out;
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
        int y, int x, float r_thresh = 10.0f) {

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

    auto t0 = std::chrono::high_resolution_clock::now();

    auto base = to_float(img);

    const int   oct  = 4;
    const int   lvl  = 6;        // scales per octave (including extras for DoG)
    const float sig0 = 1.6f;
    const float k    = std::pow(2.0f, 1.0f / (float)(lvl - 3));

    // ---- build scale-space ----
    // G[o][l]  = Gaussian images
    // D[o][l]  = DoG images (lvl-1 per octave)
    using Img2D  = std::vector<std::vector<float>>;
    using ImgSet = std::vector<Img2D>;

    std::vector<ImgSet> G, D;

    Img2D cur = gaussian_blur(base, sig0);

    for (int o = 0; o < oct; o++) {
        ImgSet g;
        g.push_back(cur);

        float sp = sig0;
        for (int l = 1; l < lvl; l++) {
            float st = sig0 * std::pow(k, (float)l);
            float sd = std::sqrt(std::max(1e-4f, st*st - sp*sp));
            g.push_back(gaussian_blur(g.back(), sd));
            sp = st;
        }
        G.push_back(g);

        ImgSet d;
        for (int l = 1; l < lvl; l++)
            d.push_back(dog(g[l-1], g[l]));
        D.push_back(d);

        // seed for next octave from s=lvl-3 (so DoG peak range is correct)
        if (o < oct-1) cur = downsample(g[lvl-3]);
    }

    // ---- detect & describe keypoints ----
    std::vector<Keypoint> kps;

    const float contrast_thresh = 0.04f / (float)(lvl - 3); // per-level threshold

    for (int o = 0; o < oct; o++) {
        int nd = (int)D[o].size();
        int h  = (int)D[o][0].size();
        int w  = (int)D[o][0][0].size();

        for (int l = 1; l < nd-1; l++)
        for (int y = 1; y < h-1;  y++)
        for (int x = 1; x < w-1;  x++) {

            float v = D[o][l][y][x];

            // cheap preliminary contrast gate
            if (std::fabs(v) < 0.5f * contrast_thresh) continue;

            // 3-D extremum test (26-neighbourhood)
            bool extremum = true;
            for (int dl = -1; dl <= 1 && extremum; dl++)
            for (int dy = -1; dy <= 1 && extremum; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                if (!dl && !dy && !dx) continue;
                float nb = D[o][l+dl][y+dy][x+dx];
                if ((v > 0 && nb >= v) || (v < 0 && nb <= v)) extremum = false;
            }
            if (!extremum) continue;

            // subpixel refinement
            float xr, yr, lr, ival;
            if (!refine(D[o], l, y, x, xr, yr, lr, ival)) continue;

            // contrast threshold on interpolated value
            if (std::fabs(ival) < contrast_thresh) continue;

            // edge rejection
            if (!edge_response_ok(D[o][l], y, x)) continue;

            // scale in image coordinates
            float scale = sig0 * std::pow(k, (float)l + lr) * std::pow(2.0f, (float)o);

            const Img2D& gimg = G[o][l];
            float xs = (float)x + xr;
            float ys = (float)y + yr;

            auto oris = orientations(gimg, xs, ys, sig0 * std::pow(k, (float)l));

            for (float ori : oris) {
                Keypoint kp;
                kp.x           = xs * std::pow(2.0f, (float)o);
                kp.y           = ys * std::pow(2.0f, (float)o);
                kp.scale       = scale;
                kp.response    = ival;
                kp.orientation = ori;
                kp.descriptor  = descriptor(gimg, xs, ys,
                                             sig0 * std::pow(k, (float)l), ori);
                kps.push_back(kp);
            }
        }
    }

    // keep top 500 by |response|
    std::sort(kps.begin(), kps.end(),
              [](const Keypoint& a, const Keypoint& b) {
                  return std::fabs(a.response) > std::fabs(b.response);
              });
    if (kps.size() > 500) kps.resize(500);

    // ---- visualise ----
    ImageData out = img;
    for (const auto& kp : kps) {
        draw_circle(out, (int)kp.x, (int)kp.y, 4, 255, 255, 255);
        int x2 = (int)(kp.x + 8.0f * std::cos(kp.orientation));
        int y2 = (int)(kp.y + 8.0f * std::sin(kp.orientation));
        draw_line(out, (int)kp.x, (int)kp.y, x2, y2, 255, 255, 255);
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    SIFTResult r;
    r.keypoints     = kps;
    r.result_image  = out;
    r.time_ms       = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.num_keypoints = (int)kps.size();
    return r;
}