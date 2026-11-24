// src/ik2/ik2_cpu.cpp - maya-ik2's CPU lane (the interface: ik2_cpu.hpp).
#include "ik2_cpu.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace ik2::cpu {

namespace {
bool avx512_ok() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    static const bool ok = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
                           __builtin_cpu_supports("avx512vl");
    return ok;
#else
    return false;
#endif
}
bool avx2_ok() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    static const bool ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
                           __builtin_cpu_supports("f16c");
    return ok;
#else
    return false;
#endif
}
}  // namespace

bool available() {
    return avx512_ok() || avx2_ok();
}

namespace {

// ggml_ue4m3_to_fp32 of every byte
struct ScaleTable {
    float v[256];
    ScaleTable() {
        for (int x = 0; x < 256; ++x) {
            if (x == 0 || x == 0x7F) {
                v[x] = 0.0f;
                continue;
            }
            const int e = (x >> 3) & 0xF, m = x & 7;
            v[x] = (e == 0 ? std::ldexp((float) m, -9) : std::ldexp(1.0f + (float) m / 8.0f, e - 7)) * 0.5f;
        }
    }
};
const ScaleTable g_scale;

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
// a 36-byte block: four UE4M3 scales, then 8 bytes a 16-value sub-block (value j < 8 the low nibble of byte j)
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
void nvfp4_rows_avx512(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    const __m128i tab = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const int nb = K / 64;
    const size_t rb = (size_t) nb * 36;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
        const float* xp = x;
        for (int i = 0; i < nb; ++i, b += 36, xp += 64) {
#define IK2_SUB(sb, acc)                                                                       \
    {                                                                                          \
        const __m128i raw = _mm_loadl_epi64((const __m128i*) (b + 4 + (sb) * 8));             \
        const __m128i lo = _mm_and_si128(raw, m4);                                             \
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);                          \
        const __m128i val = _mm_shuffle_epi8(tab, _mm_unpacklo_epi64(lo, hi));                 \
        const __m512 v = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(val));                        \
        const __m512 xs = _mm512_mul_ps(_mm512_loadu_ps(xp + (sb) * 16), _mm512_set1_ps(g_scale.v[b[sb]])); \
        acc = _mm512_fmadd_ps(v, xs, acc);                                                     \
    }
            IK2_SUB(0, acc0)
            IK2_SUB(1, acc1)
            IK2_SUB(2, acc0)
            IK2_SUB(3, acc1)
#undef IK2_SUB
        }
        out[r - r0] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    }
}

__attribute__((target("avx2,fma")))
inline float hsum256(__m256 v) {
    const __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 t = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(t, _mm_shuffle_ps(t, t, 1)));
}

// The same blocks as nvfp4_rows_avx512, eight lanes at a time: a 16-value sub-block is two __m256 halves
// (values 0..7 the low nibbles, 8..15 the high nibbles of its 8 bytes), four accumulators for the FMA latency.
__attribute__((target("avx2,fma")))
void nvfp4_rows_avx2(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    const __m128i tab = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const int nb = K / 64;
    const size_t rb = (size_t) nb * 36;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        const float* xp = x;
        for (int i = 0; i < nb; ++i, b += 36, xp += 64) {
#define IK2_SUB2(sb, accl, acch)                                                                   \
    {                                                                                              \
        const __m128i raw = _mm_loadl_epi64((const __m128i*) (b + 4 + (sb) * 8));                 \
        const __m128i lo = _mm_and_si128(raw, m4);                                                 \
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);                              \
        const __m256 vl = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_shuffle_epi8(tab, lo)));    \
        const __m256 vh = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_shuffle_epi8(tab, hi)));    \
        const __m256 s = _mm256_set1_ps(g_scale.v[b[sb]]);                                         \
        accl = _mm256_fmadd_ps(vl, _mm256_mul_ps(_mm256_loadu_ps(xp + (sb) * 16), s), accl);      \
        acch = _mm256_fmadd_ps(vh, _mm256_mul_ps(_mm256_loadu_ps(xp + (sb) * 16 + 8), s), acch);  \
    }
            IK2_SUB2(0, a0, a1)
            IK2_SUB2(1, a2, a3)
            IK2_SUB2(2, a0, a1)
            IK2_SUB2(3, a2, a3)
#undef IK2_SUB2
        }
        out[r - r0] = hsum256(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    }
}
#endif

void nvfp4_rows_scalar(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    static const int kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const int nb = K / 64;
    const size_t rb = (size_t) nb * 36;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        double acc = 0.0;
        for (int i = 0; i < nb; ++i, b += 36)
            for (int sb = 0; sb < 4; ++sb) {
                float a = 0.0f;
                for (int j = 0; j < 8; ++j) {
                    const uint8_t by = b[4 + sb * 8 + j];
                    a += kv[by & 0xF] * x[i * 64 + sb * 16 + j] + kv[by >> 4] * x[i * 64 + sb * 16 + j + 8];
                }
                acc += (double) g_scale.v[b[sb]] * a;
            }
        out[r - r0] = (float) acc;
    }
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx512f,avx512bw,avx512vl,fma,f16c")))
void q8_0_rows_avx512(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    const int nb = K / 32;
    const size_t rb = (size_t) nb * 34;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
        const float* xp = x;
        for (int i = 0; i < nb; ++i, b += 34, xp += 32) {
            uint16_t h;
            std::memcpy(&h, b, 2);
            const __m512 d = _mm512_set1_ps(_cvtsh_ss(h));
            const __m512 v0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (b + 2))));
            const __m512 v1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (b + 18))));
            acc0 = _mm512_fmadd_ps(v0, _mm512_mul_ps(_mm512_loadu_ps(xp), d), acc0);
            acc1 = _mm512_fmadd_ps(v1, _mm512_mul_ps(_mm512_loadu_ps(xp + 16), d), acc1);
        }
        out[r - r0] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    }
}

__attribute__((target("avx2,fma,f16c")))
void q8_0_rows_avx2(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    const int nb = K / 32;
    const size_t rb = (size_t) nb * 34;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        const float* xp = x;
        for (int i = 0; i < nb; ++i, b += 34, xp += 32) {
            uint16_t h;
            std::memcpy(&h, b, 2);
            const __m256 d = _mm256_set1_ps(_cvtsh_ss(h));
            const __m128i q0 = _mm_loadu_si128((const __m128i*) (b + 2));
            const __m128i q1 = _mm_loadu_si128((const __m128i*) (b + 18));
            const __m256 v0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q0));
            const __m256 v1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q0, 8)));
            const __m256 v2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1));
            const __m256 v3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q1, 8)));
            a0 = _mm256_fmadd_ps(v0, _mm256_mul_ps(_mm256_loadu_ps(xp), d), a0);
            a1 = _mm256_fmadd_ps(v1, _mm256_mul_ps(_mm256_loadu_ps(xp + 8), d), a1);
            a2 = _mm256_fmadd_ps(v2, _mm256_mul_ps(_mm256_loadu_ps(xp + 16), d), a2);
            a3 = _mm256_fmadd_ps(v3, _mm256_mul_ps(_mm256_loadu_ps(xp + 24), d), a3);
        }
        out[r - r0] = hsum256(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    }
}
#endif

float fp16_to_f32(uint16_t h) {
    const uint32_t s = (uint32_t) (h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    if (e == 0) return (s ? -1.0f : 1.0f) * std::ldexp((float) m, -24);
    if (e == 31) return m ? NAN : (s ? -INFINITY : INFINITY);
    const uint32_t bits = s | ((e + 112) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

void q8_0_rows_scalar(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    const int nb = K / 32;
    const size_t rb = (size_t) nb * 34;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        double acc = 0.0;
        for (int i = 0; i < nb; ++i, b += 34) {
            uint16_t h;
            std::memcpy(&h, b, 2);
            float a = 0.0f;
            for (int j = 0; j < 32; ++j) a += (float) (int8_t) b[2 + j] * x[i * 32 + j];
            acc += (double) fp16_to_f32(h) * a;
        }
        out[r - r0] = (float) acc;
    }
}

}  // namespace

void q8_0_rows(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (avx512_ok()) {
        q8_0_rows_avx512(W, K, r0, r1, x, out);
        return;
    }
    if (avx2_ok()) {
        q8_0_rows_avx2(W, K, r0, r1, x, out);
        return;
    }
#endif
    q8_0_rows_scalar(W, K, r0, r1, x, out);
}

void nvfp4_rows(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (avx512_ok()) {
        nvfp4_rows_avx512(W, K, r0, r1, x, out);
        return;
    }
    if (avx2_ok()) {
        nvfp4_rows_avx2(W, K, r0, r1, x, out);
        return;
    }
#endif
    nvfp4_rows_scalar(W, K, r0, r1, x, out);
}

namespace {
constexpr int kRowsA = 32, kRowsB = 64;   // rows a block: of gate and up (with their activation), of down

// An idle worker between two batches.  Spinning on PAUSE, six workers take the core's other thread and the turbo
// from the thread that drives the GPU (measured: the batches 10% shorter, the rest of a token as much longer).
// UMWAIT (WAITPKG: Alder Lake, Sapphire Rapids and later) waits in C0.2 until the epoch's cache line is written,
// which wakes it within a microsecond; the OS caps one wait (umwait_control/max_time, ~30 us).
bool waitpkg_ok() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    static const bool ok = __builtin_cpu_supports("waitpkg");
    return ok;
#else
    return false;
#endif
}
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("waitpkg")))
void umwait_epoch(const std::atomic<uint64_t>* epoch, uint64_t seen) {
    _umonitor((void*) epoch);
    if (epoch->load(std::memory_order_acquire) != seen) return;
    _umwait(0, __rdtsc() + 200000);
}
#endif
}

Pool::Pool(int n) {
    for (int i = 0; i < n; ++i)
        th_.emplace_back([this]() {
            uint64_t seen = 0;
            for (;;) {
                // spin a while before sleeping: an expert's rows come ~1 ms apart, a wake-up costs tens of us
                const bool uw = waitpkg_ok();
                for (int spin = 0; epoch_.load(std::memory_order_acquire) == seen && !stop_.load(); ++spin) {
                    if (spin < (uw ? 128 : 1 << 16)) {   // (~4 ms either way)
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
                        if (uw) {
                            umwait_epoch(&epoch_, seen);
                            continue;
                        }
#endif
                        _mm_pause();
                        continue;
                    }
                    std::unique_lock<std::mutex> lk(m_);
                    ++sleepers_;
                    cv_.wait(lk, [&]() { return epoch_.load() != seen || stop_.load(); });
                    --sleepers_;
                }
                if (stop_.load()) return;
                seen = epoch_.load();
                // (active before the jobs are looked at: wait() empties the list, then waits for no one active)
                active_.fetch_add(1);
                work();
                active_.fetch_sub(1);
            }
        });
}
Pool::~Pool() {
    {
        std::lock_guard<std::mutex> lk(m_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : th_) t.join();
}

void Pool::add(ExpertJob* job, int C, int F, float gcap, float ucap) {
    const int j = n_jobs_.load();
    if (j >= kMaxJobs) std::abort();
    if (F > scratch_F_) {   // (the first expert, or a wider model: no batch is in flight unless j > 0)
        if (j > 0) std::abort();
        scratch_.assign((size_t) kMaxJobs * 3 * F, 0.0f);
        scratch_F_ = F;
    }
    job->C = C, job->F = F, job->gcap = gcap, job->ucap = ucap;
    job->g = scratch_.data() + (size_t) j * 3 * scratch_F_;
    job->u = job->g + scratch_F_;
    job->h = job->u + scratch_F_;
    job->n_a = (F + kRowsA - 1) / kRowsA, job->n_b = (C + kRowsB - 1) / kRowsB;
    job->next_a = 0, job->done_a = 0, job->next_b = 0, job->done_b = 0;
    jobs_[j] = job;
    left_.fetch_add(1);
    n_jobs_.store(j + 1, std::memory_order_release);
    epoch_.fetch_add(1, std::memory_order_release);
    bool wake;
    {
        std::lock_guard<std::mutex> lk(m_);
        wake = sleepers_ > 0;
    }
    if (wake) cv_.notify_all();
}

// The posted experts' row blocks until none is left to take: gate's and up's rows with their activation, then -
// once all of an expert's activation is there - down's rows.
void Pool::work() {
    for (;;) {
        const int n = n_jobs_.load(std::memory_order_acquire);
        bool took = false, pending = false;
        for (int j = 0; j < n; ++j) {
            ExpertJob& J = *jobs_[j];
            const auto rows = J.q8 ? q8_0_rows : nvfp4_rows;
            while (J.next_a.load(std::memory_order_relaxed) < J.n_a) {
                const int i = J.next_a.fetch_add(1);
                if (i >= J.n_a) break;
                const int r0 = i * kRowsA, r1 = std::min(J.F, r0 + kRowsA);
                rows(J.gate, J.C, r0, r1, J.x, J.g + r0);
                rows(J.up, J.C, r0, r1, J.x, J.u + r0);
                for (int r = r0; r < r1; ++r) {
                    const float gs = J.g[r] * J.sg, us = J.u[r] * J.su;
                    J.h[r] = J.gcap * std::tanh(gs / J.gcap) * (1.0f / (1.0f + std::exp(-gs))) * J.ucap *
                             std::tanh(us / J.ucap);
                }
                J.done_a.fetch_add(1, std::memory_order_release);
                took = true;
            }
            if (J.done_a.load(std::memory_order_acquire) < J.n_a) {   // (its last blocks are still another thread's)
                pending = true;
                continue;
            }
            while (J.next_b.load(std::memory_order_relaxed) < J.n_b) {
                const int i = J.next_b.fetch_add(1);
                if (i >= J.n_b) break;
                const int r0 = i * kRowsB, r1 = std::min(J.C, r0 + kRowsB);
                rows(J.down, J.F, r0, r1, J.h, J.y + r0);
                if (J.done_b.fetch_add(1, std::memory_order_release) + 1 == J.n_b) left_.fetch_sub(1);
                took = true;
            }
        }
        if (took) continue;
        if (!pending) return;
        _mm_pause();
    }
}

void Pool::wait() {
    if (n_jobs_.load() == 0) return;
    // (the caller takes no rows: an expert's megabytes through its caches and the launches after this run cold -
    //  measured, the batches 13% shorter either way but the token no faster with the caller's help)
    while (left_.load(std::memory_order_acquire) != 0) _mm_pause();
    n_jobs_.store(0);
    while (active_.load() != 0) _mm_pause();   // no thread still looks at the jobs: they are the caller's again
}

void post_expert(Pool& pool, ExpertJob* job, int C, int F, float gcap, float ucap) {
    pool.add(job, C, F, gcap, ucap);
}

}  // namespace ik2::cpu
