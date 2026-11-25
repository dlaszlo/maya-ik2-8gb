// tests/ik2_cpu_group_test.cpp - the CPU lane's multi-input jobs (ExpertJob::n_in > 1) against its single-input
// jobs: bit for bit the same rows, NVFP4 and Q8_0, mixed batches, 1 and 6 workers; then the time of a batch of
// experts with 1 .. 16 inputs each way.  No GPU.
//   g++ -O3 -std=c++20 -pthread -I src/ik2 tests/ik2_cpu_group_test.cpp -o /tmp/ik2_cpu_group_test
#include "ik2_cpu.cpp"

#include <chrono>
#include <cstdio>
#include <random>

using namespace ik2::cpu;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    if (!available()) {
        std::puts("SKIP: no AVX2 / AVX-512");
        return 0;
    }
    const int C = 2304, F = 1152, E = 24, T = ExpertJob::kMaxIn;   // E experts (more than the caches hold), T inputs
    constexpr int R = Pool::kMaxJobs;                                 // the rows a batch at most
    std::mt19937 rng(20261125);
    long compared = 0;
    // The kernels among themselves, where the pool's blocks do not take them: 1 .. 16 inputs, row ranges that are
    // no multiple of the rows a pass, widths other than C and F - against one input and one row a call; and the
    // activation's rows past a multiple of 8 against one row a call.
    {
        std::mt19937 krng(7);
        for (bool q8 : {false, true})
            for (int K : {64, 192, 1152, 2304}) {
                const int rows = 37;
                const size_t block = q8 ? 34 : 36, rbytes = (size_t) K / (q8 ? 32 : 64) * block;
                std::vector<uint8_t> W(rows * rbytes);
                for (size_t off = 0; off < W.size(); off += block) {
                    if (q8) {
                        uint16_t scale = (uint16_t) (0x2000 + krng() % 2048);
                        std::memcpy(W.data() + off, &scale, 2);
                        for (size_t j = 2; j < block; ++j) W[off + j] = krng() & 255;
                    } else {
                        for (size_t j = 0; j < 4; ++j) W[off + j] = krng() % 127;
                        for (size_t j = 4; j < block; ++j) W[off + j] = krng() & 255;
                    }
                }
                std::vector<float> xs((size_t) T * K), ref((size_t) T * rows), got((size_t) T * rows);
                for (auto& v : xs) v = ((int) (krng() % 2001) - 1000) * 0.001f;
                for (int q = 0; q < T; ++q)
                    for (int r = 0; r < rows; ++r)
                        (q8 ? q8_0_rows : nvfp4_rows)(W.data(), K, r, r + 1, xs.data() + (size_t) q * K,
                                                      &ref[(size_t) q * rows + r]);
                for (int n = 1; n <= T; ++n)
                    for (int r0 : {0, 1, 5})
                        for (int r1 : {r0 + 1, r0 + 2, r0 + 7, r0 + 19, rows}) {
                            const float* xp[T];
                            float* op[T];
                            for (int q = 0; q < n; ++q)
                                xp[q] = xs.data() + (size_t) (T - 1 - q) * K, op[q] = got.data() + (size_t) q * rows;
                            rows_n(q8, W.data(), K, r0, r1, n, xp, op);
                            for (int q = 0; q < n; ++q)
                                if (std::memcmp(op[q], &ref[(size_t) (T - 1 - q) * rows + r0], (size_t) (r1 - r0) * 4)) {
                                    std::printf("FAIL: %s kernel, K %d, %d inputs, rows %d .. %d, input %d\n",
                                                q8 ? "Q8_0" : "NVFP4", K, n, r0, r1, q);
                                    return 1;
                                }
                            compared += (long) n * (r1 - r0);
                        }
            }
        float g[61], u[61], h[61], h1;
        for (int r = 0; r < 61; ++r) g[r] = ((int) (krng() % 2001) - 1000) * 0.02f, u[r] = ((int) (krng() % 2001) - 1000) * 0.02f;
        activation(g, u, h, 61, 0.9f, 1.1f, 10.0f, 12.0f);
        for (int r = 0; r < 61; ++r) {
            activation(g + r, u + r, &h1, 1, 0.9f, 1.1f, 10.0f, 12.0f);
            if (std::memcmp(&h[r], &h1, 4) || !std::isfinite(h1)) {
                std::printf("FAIL: the activation of row %d of 61\n", r);
                return 1;
            }
        }
    }
    for (bool q8 : {false, true}) {
        const size_t block = q8 ? 34 : 36, width = q8 ? 32 : 64;
        const size_t matrix = (size_t) C * F / width * block;
        std::vector<uint8_t> weights(3 * matrix * E);
        for (size_t off = 0; off < weights.size(); off += block) {
            if (q8) {
                uint16_t scale = (uint16_t) (0x2000 + rng() % 2048);
                std::memcpy(weights.data() + off, &scale, 2);
                for (size_t j = 2; j < block; ++j) weights[off + j] = rng() & 255;
            } else {
                for (size_t j = 0; j < 4; ++j) weights[off + j] = rng() % 127;
                for (size_t j = 4; j < block; ++j) weights[off + j] = rng() & 255;
            }
        }
        // (the inputs on cache-line boundaries, as the engine's pinned rows are: the kernels read 32 bytes a load,
        //  and at a vector's 16 mod 64 every other load would straddle two lines - a third slower at 16 inputs)
        struct Rows {
            std::vector<float> store;
            float* data() { return (float*) (((uintptr_t) store.data() + 63) & ~(uintptr_t) 63); }
        } x{std::vector<float>((size_t) T * C + 16)};
        for (size_t i = 0; i < (size_t) T * C; ++i) x.data()[i] = ((int) (rng() % 2001) - 1000) * 0.00001f;
        auto fill = [&](ExpertJob& J, int e) {
            const uint8_t* w = weights.data() + (size_t) e * 3 * matrix;
            J.gate = w, J.up = w + matrix, J.down = w + 2 * matrix;
            J.sg = 0.7f + 0.01f * e, J.su = 0.9f - 0.01f * e, J.q8 = q8;
        };
        // the single-input jobs' rows: expected[e][t]
        std::vector<float> expected((size_t) E * T * C), out((size_t) R * C);
        {
            Pool pool(6);
            for (int e = 0; e < E; ++e) {
                ExpertJob jobs[T];
                for (int t = 0; t < T; ++t) {
                    fill(jobs[t], e);
                    jobs[t].x = x.data() + (size_t) t * C, jobs[t].y = expected.data() + ((size_t) e * T + t) * C;
                    post_expert(pool, &jobs[t], C, F, 10.0f, 12.0f);
                }
                pool.wait();
            }
        }
        for (int workers : {1, 6}) {
            Pool pool(workers);
            for (int trial = 0; trial < 120; ++trial) {
                if (trial % 20 == 0) std::this_thread::sleep_for(std::chrono::milliseconds(8));
                // a batch: jobs with 1 .. T inputs (any subset of the tokens, any order) until R rows are full
                ExpertJob jobs[R];
                int row_e[R], row_t[R], rows = 0, nj = 0;
                std::fill(out.begin(), out.end(), NAN);
                while (rows < R && nj < E) {
                    const int e = nj;
                    int n = 1 + (int) (rng() % T);
                    if (trial % 3 == 0) n = 1;   // (single-input jobs through the same batch)
                    if (rows + n > R) break;
                    int perm[T];
                    for (int t = 0; t < T; ++t) perm[t] = t;
                    std::shuffle(perm, perm + T, rng);
                    ExpertJob& J = jobs[nj++];
                    fill(J, e);
                    J.n_in = n;
                    for (int q = 0; q < n; ++q) {
                        J.xs[q] = x.data() + (size_t) perm[q] * C, J.ys[q] = out.data() + (size_t) rows * C;
                        row_e[rows] = e, row_t[rows] = perm[q], ++rows;
                    }
                    J.x = J.xs[0], J.y = J.ys[0];
                }
                for (int j = 0; j < nj; ++j) post_expert(pool, &jobs[j], C, F, 10.0f, 12.0f);
                pool.wait();
                for (int r = 0; r < rows; ++r) {
                    const float* want = expected.data() + ((size_t) row_e[r] * T + row_t[r]) * C;
                    if (std::memcmp(out.data() + (size_t) r * C, want, (size_t) C * 4)) {
                        std::printf("FAIL: %s, %d workers, trial %d, row %d (expert %d, input %d)\n",
                                    q8 ? "Q8_0" : "NVFP4", workers, trial, r, row_e[r], row_t[r]);
                        return 1;
                    }
                    for (int c = 0; c < C; ++c)
                        if (!std::isfinite(out[(size_t) r * C + c])) return 1;
                }
                compared += (long) rows * C;
            }
        }
        // the time: E experts with n inputs each, every (expert, input) a job of its own against one job an expert
        if (argc > 1) {
            Pool pool(6);
            for (int n : {1, 2, 4, 8, 16}) {
                const int ne = std::min(E, R / n);
                double best[2] = {1e30, 1e30};
                for (int round = 0; round < 9; ++round)
                    for (int grouped = 0; grouped < 2; ++grouped) {
                        ExpertJob jobs[R];
                        int nj = 0, rows = 0;
                        const double t0 = now_ms();
                        for (int e = 0; e < ne; ++e) {
                            if (grouped) {
                                ExpertJob& J = jobs[nj++];
                                fill(J, e);
                                J.n_in = n;
                                for (int q = 0; q < n; ++q)
                                    J.xs[q] = x.data() + (size_t) q * C, J.ys[q] = out.data() + (size_t) rows++ * C;
                                J.x = J.xs[0], J.y = J.ys[0];
                                post_expert(pool, &J, C, F, 10.0f, 12.0f);
                            } else {
                                for (int q = 0; q < n; ++q) {
                                    ExpertJob& J = jobs[nj++];
                                    fill(J, e);
                                    J.x = x.data() + (size_t) q * C, J.y = out.data() + (size_t) rows++ * C;
                                    post_expert(pool, &J, C, F, 10.0f, 12.0f);
                                }
                            }
                        }
                        pool.wait();
                        best[grouped] = std::min(best[grouped], now_ms() - t0);
                    }
                std::printf("%s  %d experts x %d inputs: a job a pick %.2f ms, a job an expert %.2f ms (%.2fx); "
                            "an expert %.0f us -> %.0f us\n",
                            q8 ? "Q8_0 " : "NVFP4", ne, n, best[0], best[1], best[0] / best[1],
                            best[0] * 1e3 / ne, best[1] * 1e3 / ne);
            }
        }
    }
    std::printf("PASS: multi-input jobs bit for bit the single-input jobs' rows, %ld values, NVFP4 and Q8_0, 1 / 6 workers\n",
                compared);
}
