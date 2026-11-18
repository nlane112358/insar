#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using cfloat = std::complex<float>;
constexpr float PI_F     = 3.14159265358979323846f;
constexpr float TWO_PI_F = 6.28318530717958647692f;

inline float wrap_phase(float d) {
    d = std::fmod(d + PI_F, TWO_PI_F);
    if (d < 0.0f) d += TWO_PI_F;
    return d - PI_F;
}

inline bool is_power_of_two(int v) {
    return v > 0 && (v & (v - 1)) == 0;
}

// ============================================================================
// Precomputed 1D Radix-2 FFT Plan (Zero Allocations During Solve)
// ============================================================================

struct Fft1DPlan {
    int n = 0;
    std::vector<int> bit_rev;
    std::vector<cfloat> twiddles; // exp(-i * 2*pi * k / n), k = 0 .. n/2 - 1

    void init(int size) {
        n = size;
        int log_n = 0;
        while ((1 << log_n) < n) ++log_n;

        bit_rev.resize(n);
        for (int i = 0; i < n; ++i) {
            int r = 0;
            for (int b = 0; b < log_n; ++b) {
                r = (r << 1) | ((i >> b) & 1);
            }
            bit_rev[i] = r;
        }

        twiddles.resize(n / 2);
        for (int k = 0; k < n / 2; ++k) {
            double angle = -2.0 * static_cast<double>(PI_F) * k / n;
            twiddles[k] = cfloat(static_cast<float>(std::cos(angle)),
                                 static_cast<float>(std::sin(angle)));
        }
    }

    void exec(cfloat* data, bool inverse) const {
        for (int i = 0; i < n; ++i) {
            int j = bit_rev[i];
            if (i < j) std::swap(data[i], data[j]);
        }

        for (int len = 2; len <= n; len <<= 1) {
            int half = len >> 1;
            int step = n / len;
            for (int i = 0; i < n; i += len) {
                for (int j = 0; j < half; ++j) {
                    cfloat w = twiddles[j * step];
                    if (inverse) w = std::conj(w);
                    cfloat u = data[i + j];
                    cfloat v = data[i + j + half] * w;
                    data[i + j]        = u + v;
                    data[i + j + half] = u - v;
                }
            }
        }
    }
};

// Cache-blocked transpose: src (rows x cols) -> dst (cols x rows)
void transpose_blocked(const cfloat* src, cfloat* dst, int rows, int cols) {
    constexpr int TILE = 32;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int r0 = 0; r0 < rows; r0 += TILE) {
        for (int c0 = 0; c0 < cols; c0 += TILE) {
            int r_end = std::min(r0 + TILE, rows);
            int c_end = std::min(c0 + TILE, cols);
            for (int r = r0; r < r_end; ++r) {
                for (int c = c0; c < c_end; ++c) {
                    dst[static_cast<size_t>(c) * rows + r] =
                        src[static_cast<size_t>(r) * cols + c];
                }
            }
        }
    }
}

} // namespace

// ============================================================================
// CPU InSAR Engine (Persistent Scratch Buffers + Precomputed FFT Plans)
// ============================================================================

class CpuInsarEngine {
public:
    CpuInsarEngine(int width, int height)
        : W_(width), H_(height), N_(static_cast<size_t>(width) * height),
          MW_(2 * width), MH_(2 * height)
    {
        wx_.resize(N_);
        wy_.resize(N_);
        r_.resize(N_);
        z_.resize(N_);
        p_.resize(N_);
        Ap_.resize(N_);

        buf_a_.resize(static_cast<size_t>(MH_) * MW_);
        buf_b_.resize(static_cast<size_t>(MW_) * MH_);

        fft_w_.init(MW_);
        fft_h_.init(MH_);

        // Precompute cosine tables for 2D Neumann Laplacian eigenvalues
        cos_kx_.resize(MW_);
        for (int kx = 0; kx < MW_; ++kx) {
            cos_kx_[kx] = std::cos(PI_F * static_cast<float>(kx) / static_cast<float>(W_));
        }
        cos_ky_.resize(MH_);
        for (int ky = 0; ky < MH_; ++ky) {
            cos_ky_[ky] = std::cos(PI_F * static_cast<float>(ky) / static_cast<float>(H_));
        }
    }

    struct RunStats {
        double ifg_ms = 0.0;
        double unwrap_ms = 0.0;
        int iterations = 0;
        double rel_residual = 0.0;
    };

    RunStats process_pair(
        const std::vector<cfloat>& slc1,
        const std::vector<cfloat>& slc2,
        std::vector<float>& wrapped,
        std::vector<float>& coherence,
        std::vector<float>& unwrapped,
        int coh_radius = 3,
        int max_iters = 35,
        float tol = 1e-4f)
    {
        using Clock = std::chrono::steady_clock;
        RunStats stats;

        wrapped.resize(N_);
        coherence.resize(N_);
        unwrapped.assign(N_, 0.0f);

        // --- Stage 1: Fused Interferogram + Coherence ---
        auto t0 = Clock::now();
        compute_interferogram_and_coherence(slc1, slc2, wrapped, coherence, coh_radius);
        auto t1 = Clock::now();

        // --- Stage 2: Coherence-Weighted PCG Phase Unwrapping ---
        build_weights_and_rhs(wrapped, coherence, 1e-4f);

        double b_norm = std::sqrt(dot_product(r_, r_));
        if (b_norm > 1e-12) {
            apply_poisson_preconditioner(r_, z_);
            std::copy(z_.begin(), z_.end(), p_.begin());
            double rz_old = dot_product(r_, z_);

            for (int iter = 1; iter <= max_iters; ++iter) {
                apply_weighted_laplacian(p_, Ap_);
                double pAp = dot_product(p_, Ap_);
                if (pAp <= 1e-18) break;

                float alpha = static_cast<float>(rz_old / pAp);
                #pragma omp parallel for schedule(static)
                for (int64_t i = 0; i < static_cast<int64_t>(N_); ++i) {
                    unwrapped[i] += alpha * p_[i];
                    r_[i]        -= alpha * Ap_[i];
                }

                double r_norm = std::sqrt(dot_product(r_, r_));
                stats.iterations = iter;
                stats.rel_residual = r_norm / b_norm;
                if (stats.rel_residual < tol) break;

                apply_poisson_preconditioner(r_, z_);
                double rz_new = dot_product(r_, z_);
                float beta = static_cast<float>(rz_new / (rz_old + 1e-30));

                #pragma omp parallel for schedule(static)
                for (int64_t i = 0; i < static_cast<int64_t>(N_); ++i) {
                    p_[i] = z_[i] + beta * p_[i];
                }
                rz_old = rz_new;
            }
        }

        auto t2 = Clock::now();
        stats.ifg_ms    = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats.unwrap_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        return stats;
    }

private:
    void compute_interferogram_and_coherence(
        const std::vector<cfloat>& slc1,
        const std::vector<cfloat>& slc2,
        std::vector<float>& wrapped,
        std::vector<float>& coh,
        int radius) const
    {
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < H_; ++y) {
            int y0 = std::max(0, y - radius);
            int y1 = std::min(H_ - 1, y + radius);
            for (int x = 0; x < W_; ++x) {
                int x0 = std::max(0, x - radius);
                int x1 = std::min(W_ - 1, x + radius);

                float sum_re = 0.0f, sum_im = 0.0f;
                float pow1 = 0.0f, pow2 = 0.0f;

                for (int j = y0; j <= y1; ++j) {
                    size_t row = static_cast<size_t>(j) * W_;
                    for (int i = x0; i <= x1; ++i) {
                        const cfloat a = slc1[row + i];
                        const cfloat b = slc2[row + i];
                        sum_re += a.real() * b.real() + a.imag() * b.imag();
                        sum_im += a.imag() * b.real() - a.real() * b.imag();
                        pow1   += a.real() * a.real() + a.imag() * a.imag();
                        pow2   += b.real() * b.real() + b.imag() * b.imag();
                    }
                }

                size_t idx = static_cast<size_t>(y) * W_ + x;
                wrapped[idx] = std::atan2(sum_im, sum_re);
                float denom  = std::sqrt(pow1 * pow2) + 1e-12f;
                coh[idx]     = std::clamp(std::hypot(sum_re, sum_im) / denom, 0.0f, 1.0f);
            }
        }
    }

    void build_weights_and_rhs(
        const std::vector<float>& psi,
        const std::vector<float>& coh,
        float min_weight)
    {
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < H_; ++y) {
            for (int x = 0; x < W_; ++x) {
                size_t idx = static_cast<size_t>(y) * W_ + x;
                float c_here = coh[idx];

                float w_right = 0.0f, dx_right = 0.0f;
                if (x + 1 < W_) {
                    float c = std::min(c_here, coh[idx + 1]);
                    w_right  = std::max(min_weight, c * c * c);
                    dx_right = wrap_phase(psi[idx + 1] - psi[idx]);
                }
                wx_[idx] = w_right;

                float w_down = 0.0f, dy_down = 0.0f;
                if (y + 1 < H_) {
                    float c = std::min(c_here, coh[idx + W_]);
                    w_down  = std::max(min_weight, c * c * c);
                    dy_down = wrap_phase(psi[idx + W_] - psi[idx]);
                }
                wy_[idx] = w_down;

                float w_left = 0.0f, dx_left = 0.0f;
                if (x > 0) {
                    float c = std::min(coh[idx - 1], c_here);
                    w_left  = std::max(min_weight, c * c * c);
                    dx_left = wrap_phase(psi[idx] - psi[idx - 1]);
                }

                float w_up = 0.0f, dy_up = 0.0f;
                if (y > 0) {
                    float c = std::min(coh[idx - W_], c_here);
                    w_up  = std::max(min_weight, c * c * c);
                    dy_up = wrap_phase(psi[idx] - psi[idx - W_]);
                }

                r_[idx] = (-w_right * dx_right + w_left * dx_left) +
                          (-w_down  * dy_down  + w_up   * dy_up);
            }
        }
    }

    void apply_weighted_laplacian(const std::vector<float>& p, std::vector<float>& Ap) const {
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < H_; ++y) {
            for (int x = 0; x < W_; ++x) {
                size_t idx = static_cast<size_t>(y) * W_ + x;
                float center = p[idx];
                float acc = 0.0f;

                if (x + 1 < W_) acc += wx_[idx]      * (center - p[idx + 1]);
                if (x > 0)      acc += wx_[idx - 1]  * (center - p[idx - 1]);
                if (y + 1 < H_) acc += wy_[idx]      * (center - p[idx + W_]);
                if (y > 0)      acc += wy_[idx - W_] * (center - p[idx - W_]);

                Ap[idx] = acc;
            }
        }
    }

    // Exact Neumann Poisson solver via half-sample mirrored 2D FFT
    void apply_poisson_preconditioner(const std::vector<float>& in, std::vector<float>& out) {
        // 1. Half-sample mirror pad (H x W -> 2H x 2W) into buf_a_
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < H_; ++y) {
            size_t row_top = static_cast<size_t>(y) * MW_;
            size_t row_bot = static_cast<size_t>(MH_ - 1 - y) * MW_;
            for (int x = 0; x < W_; ++x) {
                cfloat v(in[static_cast<size_t>(y) * W_ + x], 0.0f);
                buf_a_[row_top + x]                = v;
                buf_a_[row_top + (MW_ - 1 - x)]    = v;
                buf_a_[row_bot + x]                = v;
                buf_a_[row_bot + (MW_ - 1 - x)]    = v;
            }
        }

        // 2. Forward 1D FFT along rows (x -> kx)
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < MH_; ++y) {
            fft_w_.exec(&buf_a_[static_cast<size_t>(y) * MW_], false);
        }

        // 3. Cache-blocked transpose (MH x MW -> MW x MH)
        transpose_blocked(buf_a_.data(), buf_b_.data(), MH_, MW_);

        // 4. Forward 1D FFT along columns (y -> ky), spectral divide, and Inverse 1D FFT (ky -> y)
        const float inv_area = 1.0f / (static_cast<float>(MW_) * static_cast<float>(MH_));
        #pragma omp parallel for schedule(static)
        for (int kx = 0; kx < MW_; ++kx) {
            cfloat* col = &buf_b_[static_cast<size_t>(kx) * MH_];
            fft_h_.exec(col, false);

            float cx = cos_kx_[kx];
            for (int ky = 0; ky < MH_; ++ky) {
                if (kx == 0 && ky == 0) {
                    col[0] = cfloat(0.0f, 0.0f);
                } else {
                    float eig = 4.0f - 2.0f * cx - 2.0f * cos_ky_[ky];
                    col[ky] *= (inv_area / eig);
                }
            }

            fft_h_.exec(col, true);
        }

        // 5. Transpose back (MW x MH -> MH x MW)
        transpose_blocked(buf_b_.data(), buf_a_.data(), MW_, MH_);

        // 6. Inverse 1D FFT along rows (kx -> x) and crop top-left H x W quadrant
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < H_; ++y) {
            cfloat* row = &buf_a_[static_cast<size_t>(y) * MW_];
            fft_w_.exec(row, true);
            size_t out_row = static_cast<size_t>(y) * W_;
            for (int x = 0; x < W_; ++x) {
                out[out_row + x] = row[x].real();
            }
        }
    }

    double dot_product(const std::vector<float>& a, const std::vector<float>& b) const {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum) schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(N_); ++i) {
            sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        }
        return sum;
    }

    int W_, H_;
    size_t N_;
    int MW_, MH_;

    std::vector<float> wx_, wy_, r_, z_, p_, Ap_;
    std::vector<cfloat> buf_a_, buf_b_;
    std::vector<float> cos_kx_, cos_ky_;
    Fft1DPlan fft_w_, fft_h_;
};

// ============================================================================
// Synthetic Sentinel-1 Scene Generator & Binary I/O
// ============================================================================

void generate_synthetic_scene(
    int W, int H,
    std::vector<cfloat>& slc1,
    std::vector<cfloat>& slc2,
    std::vector<float>& true_phase_rad,
    std::vector<float>& true_coh)
{
    size_t N = static_cast<size_t>(W) * static_cast<size_t>(H);
    slc1.resize(N);
    slc2.resize(N);
    true_phase_rad.resize(N);
    true_coh.resize(N);

    std::mt19937 rng(42);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    // Sentinel-1 C-band wavelength: 55.4658 mm (1 full 2*pi fringe = 27.73 mm LOS)
    constexpr float LAMBDA_MM = 55.4658f;
    constexpr float MM_TO_RAD = (4.0f * PI_F) / LAMBDA_MM;

    for (int y = 0; y < H; ++y) {
        float yn = (static_cast<float>(y) / H) - 0.5f;
        for (int x = 0; x < W; ++x) {
            float xn = (static_cast<float>(x) / W) - 0.5f;
            size_t idx = static_cast<size_t>(y) * W + x;

            float r1 = (xn + 0.12f) * (xn + 0.12f) / 0.018f + (yn - 0.05f) * (yn - 0.05f) / 0.014f;
            float r2 = (xn - 0.20f) * (xn - 0.20f) / 0.008f + (yn + 0.18f) * (yn + 0.18f) / 0.010f;
            float disp_mm = -95.0f * std::exp(-r1) - 40.0f * std::exp(-r2) + 12.0f * xn - 8.0f * yn;
            float phi = disp_mm * MM_TO_RAD;
            true_phase_rad[idx] = phi;

            // Simulated decorrelated water/forest patch in upper-right
            float r_lake = (xn - 0.15f) * (xn - 0.15f) / 0.012f + (yn + 0.15f) * (yn + 0.15f) / 0.012f;
            float rho = (r_lake < 1.0f) ? 0.15f : 0.90f;
            true_coh[idx] = rho;

            float a_re = gauss(rng), a_im = gauss(rng);
            float n_re = gauss(rng), n_im = gauss(rng);

            float c = std::cos(phi);
            float s = std::sin(phi);
            float rot_re = a_re * c + a_im * s;
            float rot_im = a_im * c - a_re * s;
            float noise_scale = std::sqrt(1.0f - rho * rho);

            slc1[idx] = cfloat(a_re, a_im);
            slc2[idx] = cfloat(rho * rot_re + noise_scale * n_re,
                               rho * rot_im + noise_scale * n_im);
        }
    }
}

bool read_complex_bin(const std::string& path, size_t count, std::vector<cfloat>& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    out.resize(count);
    in.read(reinterpret_cast<char*>(out.data()), count * sizeof(cfloat));
    return static_cast<size_t>(in.gcount()) == count * sizeof(cfloat);
}

void write_float_bin(const std::string& path, const std::vector<float>& data) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}

int main(int argc, char** argv) {
    int W = 1024;
    int H = 1024;
    int max_iters = 35;
    bool dump_files = false;
    std::string slc1_path, slc2_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--width" && i + 1 < argc)      W = std::stoi(argv[++i]);
        else if (arg == "--height" && i + 1 < argc) H = std::stoi(argv[++i]);
        else if (arg == "--iters" && i + 1 < argc)  max_iters = std::stoi(argv[++i]);
        else if (arg == "--slc1" && i + 1 < argc)   slc1_path = argv[++i];
        else if (arg == "--slc2" && i + 1 < argc)   slc2_path = argv[++i];
        else if (arg == "--dump")                   dump_files = true;
    }

    if (!is_power_of_two(W) || !is_power_of_two(H)) {
        std::fprintf(stderr, "Error: Built-in radix-2 FFT requires power-of-2 dimensions "
                             "(e.g., 512, 1024, 2048). Got %d x %d.\n", W, H);
        return EXIT_FAILURE;
    }

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    std::printf("CPU Backend (%d OpenMP threads) | Scene: %d x %d (%.2f Mpx)\n",
                threads, W, H, (W * H) / 1e6);

    std::vector<cfloat> slc1, slc2;
    std::vector<float> true_phi, true_coh;
    bool synthetic = slc1_path.empty() || slc2_path.empty();

    if (synthetic) {
        generate_synthetic_scene(W, H, slc1, slc2, true_phi, true_coh);
    } else {
        size_t N = static_cast<size_t>(W) * H;
        if (!read_complex_bin(slc1_path, N, slc1) || !read_complex_bin(slc2_path, N, slc2)) {
            std::fprintf(stderr, "Error: Failed to read complex float32 SLC files.\n");
            return EXIT_FAILURE;
        }
    }

    CpuInsarEngine engine(W, H);
    std::vector<float> wrapped, coh, unwrapped;

    auto stats = engine.process_pair(slc1, slc2, wrapped, coh, unwrapped, 3, max_iters, 1e-4f);

    std::printf("------------------------------------------------------------\n");
    std::printf("Stage 1 (Interferogram + 7x7 Coherence): %8.2f ms\n", stats.ifg_ms);
    std::printf("Stage 2 (Weighted PCG Unwrap, %2d iters): %8.2f ms (res=%.2e)\n",
                stats.iterations, stats.unwrap_ms, stats.rel_residual);
    std::printf("Total CPU Compute Time:                  %8.2f ms\n",
                stats.ifg_ms + stats.unwrap_ms);

    if (synthetic) {
        constexpr float RAD_TO_MM = 55.4658f / (4.0f * PI_F);
        double mean_diff = 0.0;
        size_t valid_count = 0;
        for (size_t i = 0; i < unwrapped.size(); ++i) {
            if (true_coh[i] > 0.5f) {
                mean_diff += (unwrapped[i] - true_phi[i]);
                ++valid_count;
            }
        }
        mean_diff /= static_cast<double>(valid_count);

        double sq_err_mm = 0.0;
        for (size_t i = 0; i < unwrapped.size(); ++i) {
            if (true_coh[i] > 0.5f) {
                double err_mm = ((unwrapped[i] - true_phi[i]) - mean_diff) * RAD_TO_MM;
                sq_err_mm += err_mm * err_mm;
            }
        }
        double rmse_mm = std::sqrt(sq_err_mm / static_cast<double>(valid_count));
        std::printf("Coherent-Area LOS Accuracy (C-band):     %8.2f mm RMSE\n", rmse_mm);
    }
    std::printf("------------------------------------------------------------\n");

    if (dump_files) {
        write_float_bin("wrapped_phase.f32", wrapped);
        write_float_bin("coherence.f32", coh);
        write_float_bin("unwrapped_phase.f32", unwrapped);
        if (synthetic) write_float_bin("true_phase.f32", true_phi);
        std::printf("Wrote binary float32 rasters for visualization.\n");
    }

    return 0;
}
