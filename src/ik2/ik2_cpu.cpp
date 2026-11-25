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

// ---- The AVX2 kernels ----
// A row's weights are decoded to F32 - exactly: an NVFP4 value times its sub-block's scale has 8 significant bits
// at most, a Q8_0 value times its block's FP16 scale 19 - and a product is one FMA.  Per input and row, with two
// accumulators of 8 lanes (the 8-value groups of the row alternate between them):
//     acc[(k >> 3) & 1][k & 7] = fma(w[k], x[k], acc[(k >> 3) & 1][k & 7])   for k = 0 .. K - 1 in this order,
//     the row = hsum256(acc[0] + acc[1]).
// Every kernel below - any number of inputs, one or several rows a pass, the weights decoded in registers or
// through a buffer, the accumulators in registers all along or parked in memory between two column steps - does
// exactly these operations in this order for an (input, row): they agree bit for bit.
// (The scale is no longer multiplied into the input, as the AVX-512 kernels above still do: one multiplication
//  a decoded value for all the inputs, instead of one an input.  The sums differ from those kernels' by a
//  rounding of every product.)
#define IK2_AVX2 __attribute__((target("avx2,fma,f16c")))
#define IK2_AVX2_INLINE __attribute__((target("avx2,fma,f16c"), always_inline)) inline

// 8 int8 values times a scale, as F32
IK2_AVX2_INLINE __m256 i8x8_ps(const uint8_t* p, __m256 s) {
    return _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) p))), s);
}

// A format: its blocks of kBlock values in kBytes bytes; open() a block, then pair() its 16-value groups in order
// (wl the values 0 .. 7 of the group, wh 8 .. 15).  kRows1 / kRows2: the rows a pass of one input / of two (below).
struct Nvfp4 {
    static constexpr int kBlock = 64, kBytes = 36, kRows1 = 4, kRows2 = 2;
    // the block's 64 values as int8: t[sb * 8 + j] value j < 8 of sub-block sb, t[32 + sb * 8 + j] value 8 + j
    struct Blk {
        alignas(32) uint8_t t[64];
    };
    static IK2_AVX2_INLINE void open(const uint8_t* b, Blk& s) {
        const __m256i tab = _mm256_broadcastsi128_si256(
            _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12));
        const __m256i m4 = _mm256_set1_epi8(0x0F);
        const __m256i raw = _mm256_loadu_si256((const __m256i*) (b + 4));
        _mm256_store_si256((__m256i*) s.t, _mm256_shuffle_epi8(tab, _mm256_and_si256(raw, m4)));
        _mm256_store_si256((__m256i*) (s.t + 32),
                           _mm256_shuffle_epi8(tab, _mm256_and_si256(_mm256_srli_epi16(raw, 4), m4)));
        // (t really in memory: the eight VPMOVSXBD below then read it with their load - the store forwards -
        //  instead of the compiler's shuffles of the two registers, which crowd the ports the FMAs need.
        //  Measured: one input 21% faster.)
        __asm__ volatile("" : : "r"(s.t) : "memory");
    }
    static IK2_AVX2_INLINE void pair(const Blk& s, const uint8_t* b, int g, __m256& wl, __m256& wh) {
        const __m256 sc = _mm256_broadcast_ss(&g_scale.v[b[g]]);
        wl = i8x8_ps(s.t + g * 8, sc);
        wh = i8x8_ps(s.t + 32 + g * 8, sc);
    }
};
struct Q8 {
    static constexpr int kBlock = 32, kBytes = 34, kRows1 = 1, kRows2 = 1;
    struct Blk {
        __m256 d;
    };
    static IK2_AVX2_INLINE void open(const uint8_t* b, Blk& s) {
        uint16_t h;
        std::memcpy(&h, b, 2);
        s.d = _mm256_set1_ps(_cvtsh_ss(h));
    }
    static IK2_AVX2_INLINE void pair(const Blk& s, const uint8_t* b, int g, __m256& wl, __m256& wh) {
        wl = i8x8_ps(b + 2 + g * 16, s.d);
        wh = i8x8_ps(b + 10 + g * 16, s.d);
    }
};

// ---- Up to four inputs: the weights decoded in registers ----
// R consecutive rows (the first at b0, rb bytes a row, nb blocks) against N inputs: outs[q][j] = row j . xs[q].
// N * R * 2 FMA chains - 8 fill the two FMA ports, so one input or two would take 4 or 2 rows a pass.  They do
// for NVFP4 (one input 0.76 instead of 0.98 ns a 16 values with the expert in cache, and no slower from memory);
// Q8_0's rows are 2448 bytes, and several of them read side by side come slower from memory than one after
// the other (measured, 6 workers, us an expert: one input 191 / 199 / 206 with 1 / 2 / 4 rows a pass, two inputs
// 198 / 206 with 1 / 2) - one row.
template <class Fmt, int N, int R>
IK2_AVX2_INLINE void rows_core(const uint8_t* b0, size_t rb, int nb, const float* const* xs, float* const* outs) {
    const float* x[N];
    const uint8_t* b[R];
    __m256 lo[R][N], hi[R][N];
    for (int q = 0; q < N; ++q) x[q] = xs[q];
    for (int j = 0; j < R; ++j) {
        b[j] = b0 + (size_t) j * rb;
        for (int q = 0; q < N; ++q) lo[j][q] = hi[j][q] = _mm256_setzero_ps();
    }
    size_t k = 0;
    for (int i = 0; i < nb; ++i) {
        typename Fmt::Blk blk[R];
        for (int j = 0; j < R; ++j) Fmt::open(b[j], blk[j]);
        for (int g = 0; g < Fmt::kBlock / 16; ++g, k += 16)
            for (int j = 0; j < R; ++j) {
                __m256 wl, wh;
                Fmt::pair(blk[j], b[j], g, wl, wh);
                for (int q = 0; q < N; ++q) {
                    lo[j][q] = _mm256_fmadd_ps(wl, _mm256_loadu_ps(x[q] + k), lo[j][q]);
                    hi[j][q] = _mm256_fmadd_ps(wh, _mm256_loadu_ps(x[q] + k + 8), hi[j][q]);
                }
            }
        for (int j = 0; j < R; ++j) b[j] += Fmt::kBytes;
    }
    for (int j = 0; j < R; ++j)
        for (int q = 0; q < N; ++q) outs[q][j] = hsum256(_mm256_add_ps(lo[j][q], hi[j][q]));
}

// outs[q][r - r0] = W[r] . xs[q] for rows r0 .. r1 - 1 and N inputs, R rows a pass (then the rows left one a pass)
template <class Fmt, int N, int R>
IK2_AVX2 void rows_avx2(const uint8_t* W, int K, int r0, int r1, const float* const* xs, float* const* outs) {
    const int nb = K / Fmt::kBlock;
    const size_t rb = (size_t) nb * Fmt::kBytes;
    float* o[N];
    int r = r0;
    for (; r + R <= r1; r += R) {
        for (int q = 0; q < N; ++q) o[q] = outs[q] + (r - r0);
        rows_core<Fmt, N, R>(W + (size_t) r * rb, rb, nb, xs, o);
    }
    for (; r < r1; ++r) {
        for (int q = 0; q < N; ++q) o[q] = outs[q] + (r - r0);
        rows_core<Fmt, N, 1>(W + (size_t) r * rb, rb, nb, xs, o);
    }
}

// ---- More than four inputs: blocked for L1 ----
// kRun rows, kCols columns a step: that piece of the rows decoded once into w (kRun x kCols F32), then the inputs
// four a pass over it - the decoding is shared by all the inputs.  The accumulators of every (row, input) wait in
// acc between two column steps: the FMAs of an (input, row) are rows_core's, in its order.  w, acc and the
// columns of a pass's four inputs stay in L1 (16 + 16 + 4 KB at 16 inputs).
constexpr int kRun = 16, kCols = 256;   // (kCols: whole blocks of either format)

// kn columns from k0 of R decoded rows (w0, ws values apart) against N inputs; the accumulators of row j and
// input q at acc + j * as + q * 16 (lo, then hi)
template <int N, int R>
IK2_AVX2_INLINE void dot_chunk(const float* w0, size_t ws, int kn, const float* const* xs, size_t k0, float* acc,
                               size_t as) {
    const float* x[N];
    __m256 lo[R][N], hi[R][N];
    for (int q = 0; q < N; ++q) x[q] = xs[q] + k0;
    for (int j = 0; j < R; ++j)
        for (int q = 0; q < N; ++q) {
            lo[j][q] = _mm256_load_ps(acc + j * as + q * 16);
            hi[j][q] = _mm256_load_ps(acc + j * as + q * 16 + 8);
        }
#define IK2_DOT_STEP(k)                                                                       \
    for (int j = 0; j < R; ++j) {                                                            \
        const __m256 wl = _mm256_load_ps(w0 + j * ws + (k));                                 \
        const __m256 wh = _mm256_load_ps(w0 + j * ws + (k) + 8);                             \
        for (int q = 0; q < N; ++q) {                                                        \
            lo[j][q] = _mm256_fmadd_ps(wl, _mm256_loadu_ps(x[q] + (k)), lo[j][q]);           \
            hi[j][q] = _mm256_fmadd_ps(wh, _mm256_loadu_ps(x[q] + (k) + 8), hi[j][q]);       \
        }                                                                                    \
    }
    // (64 values a turn: the addresses are a register and a constant, an FMA and its load stay one micro-op)
    int k = 0;
    for (; k + 64 <= kn; k += 64) {
        IK2_DOT_STEP(k)
        IK2_DOT_STEP(k + 16)
        IK2_DOT_STEP(k + 32)
        IK2_DOT_STEP(k + 48)
    }
    for (; k < kn; k += 16) IK2_DOT_STEP(k)
#undef IK2_DOT_STEP
    for (int j = 0; j < R; ++j)
        for (int q = 0; q < N; ++q) {
            _mm256_store_ps(acc + j * as + q * 16, lo[j][q]);
            _mm256_store_ps(acc + j * as + q * 16 + 8, hi[j][q]);
        }
}
// (R rows a pass, then the rows left one a pass)
template <int N, int R>
IK2_AVX2_INLINE void dot_chunks(const float* w, int nr, int kn, const float* const* xs, size_t k0, float* acc,
                                size_t as) {
    int j = 0;
    for (; j + R <= nr; j += R) dot_chunk<N, R>(w + (size_t) j * kn, kn, kn, xs, k0, acc + j * as, as);
    for (; j < nr; ++j) dot_chunk<N, 1>(w + (size_t) j * kn, kn, kn, xs, k0, acc + j * as, as);
}

template <class Fmt>
IK2_AVX2 void rows_blocked_avx2(const uint8_t* W, int K, int r0, int r1, int n, const float* const* xs,
                                float* const* outs) {
    alignas(64) float w[kRun * kCols];
    alignas(64) float acc[kRun * ExpertJob::kMaxIn * 16];
    const size_t rb = (size_t) (K / Fmt::kBlock) * Fmt::kBytes, as = (size_t) n * 16;
    for (int ra = r0; ra < r1; ra += kRun) {
        const int nr = std::min(kRun, r1 - ra);
        std::memset(acc, 0, (size_t) nr * as * sizeof(float));
        for (int k0 = 0; k0 < K; k0 += kCols) {
            const int kn = std::min(kCols, K - k0);
            // (the next step's pieces of the rows - short and a row apart - are asked for now and come from memory
            //  while this step is computed.  Measured: Q8_0 with 5 .. 8 inputs 3 - 4% faster, the rest the same.)
            {
                int kk = k0 + kCols, rr = ra;
                if (kk >= K) kk = 0, rr = ra + kRun;
                const size_t off = (size_t) (kk / Fmt::kBlock) * Fmt::kBytes;
                const size_t len = (size_t) (std::min(kCols, K - kk) / Fmt::kBlock) * Fmt::kBytes;
                for (int r = rr; r < std::min(r1, rr + kRun); ++r) {
                    const char* const p = (const char*) W + (size_t) r * rb + off;
                    for (size_t o = 0; o < len; o += 64) _mm_prefetch(p + o, _MM_HINT_T0);
                    _mm_prefetch(p + len - 1, _MM_HINT_T0);
                }
            }
            float* wp = w;
            for (int j = 0; j < nr; ++j) {
                const uint8_t* b = W + (size_t) (ra + j) * rb + (size_t) (k0 / Fmt::kBlock) * Fmt::kBytes;
                for (int i = 0; i < kn / Fmt::kBlock; ++i, b += Fmt::kBytes) {
                    typename Fmt::Blk blk;
                    Fmt::open(b, blk);
                    for (int g = 0; g < Fmt::kBlock / 16; ++g, wp += 16) {
                        __m256 wl, wh;
                        Fmt::pair(blk, b, g, wl, wh);
                        _mm256_store_ps(wp, wl);
                        _mm256_store_ps(wp + 8, wh);
                    }
                }
            }
            // (the inputs left after the fours: with 2 or 4 rows a pass, for the 8 chains)
            for (int q = 0; q < n; q += 4) switch (std::min(4, n - q)) {
                case 1: dot_chunks<1, 4>(w, nr, kn, xs + q, k0, acc + q * 16, as); break;
                case 2: dot_chunks<2, 2>(w, nr, kn, xs + q, k0, acc + q * 16, as); break;
                case 3: dot_chunks<3, 1>(w, nr, kn, xs + q, k0, acc + q * 16, as); break;
                default: dot_chunks<4, 1>(w, nr, kn, xs + q, k0, acc + q * 16, as); break;
                }
        }
        for (int j = 0; j < nr; ++j)
            for (int q = 0; q < n; ++q)
                outs[q][ra - r0 + j] = hsum256(_mm256_add_ps(_mm256_load_ps(acc + j * as + q * 16),
                                                             _mm256_load_ps(acc + j * as + q * 16 + 8)));
    }
}

// Rows r0 .. r1 - 1 against n inputs
template <class Fmt>
IK2_AVX2 void rows_n_avx2(const uint8_t* W, int K, int r0, int r1, int n, const float* const* xs,
                          float* const* outs) {
    switch (n) {
    case 1: rows_avx2<Fmt, 1, Fmt::kRows1>(W, K, r0, r1, xs, outs); return;
    case 2: rows_avx2<Fmt, 2, Fmt::kRows2>(W, K, r0, r1, xs, outs); return;
    case 3: rows_avx2<Fmt, 3, 1>(W, K, r0, r1, xs, outs); return;
    case 4: rows_avx2<Fmt, 4, 1>(W, K, r0, r1, xs, outs); return;
    }
    rows_blocked_avx2<Fmt>(W, K, r0, r1, n, xs, outs);
}

// ---- The activation, 8 rows a step ----
// e^x - 1 for x in [-87, 87]: x = n ln 2 + r, |r| <= ln 2 / 2; e^r - 1 = r + r^2 P(r) (Cephes' expf polynomial);
// 2^n (e^r - 1) + (2^n - 1) - no cancellation at small x (n = 0: the polynomial itself).
IK2_AVX2_INLINE __m256 expm1_avx2(__m256 x) {
    const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), r);
    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
    const __m256 q = _mm256_fmadd_ps(_mm256_mul_ps(r, r), p, r);
    const __m256 two_n = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23));
    return _mm256_fmadd_ps(two_n, q, _mm256_sub_ps(two_n, _mm256_set1_ps(1.0f)));
}
// tanh z = sign(z) m / (m + 2), m = e^(2 |z|) - 1
IK2_AVX2_INLINE __m256 tanh_avx2(__m256 z) {
    const __m256 sign = _mm256_and_ps(z, _mm256_set1_ps(-0.0f));
    const __m256 a = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), z);
    const __m256 m = expm1_avx2(_mm256_min_ps(_mm256_set1_ps(87.0f), _mm256_add_ps(a, a)));   // (a NaN stays one)
    return _mm256_or_ps(_mm256_div_ps(m, _mm256_add_ps(m, _mm256_set1_ps(2.0f))), sign);
}
// h = gcap tanh(gs / gcap) sigmoid(gs) ucap tanh(us / ucap), gs = g sg, us = u su;  sigmoid(gs) = 1 / (2 + (e^-gs - 1))
IK2_AVX2_INLINE __m256 act_avx2(__m256 g, __m256 u, __m256 sg, __m256 su, __m256 gcap, __m256 ucap) {
    const __m256 gs = _mm256_mul_ps(g, sg), us = _mm256_mul_ps(u, su);
    const __m256 tg = tanh_avx2(_mm256_div_ps(gs, gcap)), tu = tanh_avx2(_mm256_div_ps(us, ucap));
    const __m256 ng = _mm256_min_ps(
        _mm256_set1_ps(87.0f), _mm256_max_ps(_mm256_set1_ps(-87.0f), _mm256_xor_ps(gs, _mm256_set1_ps(-0.0f))));
    const __m256 sig = _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(_mm256_set1_ps(2.0f), expm1_avx2(ng)));
    return _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(gcap, tg), sig), ucap), tu);
}
// (the one function of every job, whatever its inputs: a row's value does not depend on the rows around it -
//  the rows past a multiple of 8 go through the same lanes)
IK2_AVX2 void activation_avx2(const float* g, const float* u, float* h, int n, float sg, float su, float gcap,
                              float ucap) {
    const __m256 vsg = _mm256_set1_ps(sg), vsu = _mm256_set1_ps(su), vgc = _mm256_set1_ps(gcap),
                 vuc = _mm256_set1_ps(ucap);
    int r = 0;
    for (; r + 8 <= n; r += 8)
        _mm256_storeu_ps(h + r, act_avx2(_mm256_loadu_ps(g + r), _mm256_loadu_ps(u + r), vsg, vsu, vgc, vuc));
    if (r < n) {
        float tg[8] = {}, tu[8] = {}, th[8];
        std::memcpy(tg, g + r, (size_t) (n - r) * sizeof(float));
        std::memcpy(tu, u + r, (size_t) (n - r) * sizeof(float));
        _mm256_storeu_ps(th, act_avx2(_mm256_loadu_ps(tg), _mm256_loadu_ps(tu), vsg, vsu, vgc, vuc));
        std::memcpy(h + r, th, (size_t) (n - r) * sizeof(float));
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
        rows_n_avx2<Q8>(W, K, r0, r1, 1, &x, &out);
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
        rows_n_avx2<Nvfp4>(W, K, r0, r1, 1, &x, &out);
        return;
    }
#endif
    nvfp4_rows_scalar(W, K, r0, r1, x, out);
}

namespace {
// The same rows against n inputs: outs[q][r - r0] = W[r] . xs[q], bit for bit what nvfp4_rows / q8_0_rows give
// for xs[q].  AVX2: the weights read and decoded once for all of them.  (AVX-512 and scalar: the single-input
// kernel an input - a row block is still in the core's cache the next time.)
void rows_n(bool q8, const uint8_t* W, int K, int r0, int r1, int n, const float* const* xs, float* const* outs) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (!avx512_ok() && avx2_ok()) {
        if (q8)
            rows_n_avx2<Q8>(W, K, r0, r1, n, xs, outs);
        else
            rows_n_avx2<Nvfp4>(W, K, r0, r1, n, xs, outs);
        return;
    }
#endif
    for (int q = 0; q < n; ++q) (q8 ? q8_0_rows : nvfp4_rows)(W, K, r0, r1, xs[q], outs[q]);
}

// h[r] = gcap tanh(gs / gcap) sigmoid(gs) ucap tanh(us / ucap), gs = g[r] sg, us = u[r] su, for n rows
void activation(const float* g, const float* u, float* h, int n, float sg, float su, float gcap, float ucap) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (avx2_ok()) {
        activation_avx2(g, u, h, n, sg, su, gcap, ucap);
        return;
    }
#endif
    for (int r = 0; r < n; ++r) {
        const float gs = g[r] * sg, us = u[r] * su;
        h[r] = gcap * std::tanh(gs / gcap) * (1.0f / (1.0f + std::exp(-gs))) * ucap * std::tanh(us / ucap);
    }
}
}  // namespace

void nvfp4_rows2(const uint8_t* W, int K, int r0, int r1, const float* x0, const float* x1, float* out0, float* out1) {
    const float* xs[2] = {x0, x1};
    float* outs[2] = {out0, out1};
    rows_n(false, W, K, r0, r1, 2, xs, outs);
}

void q8_0_rows2(const uint8_t* W, int K, int r0, int r1, const float* x0, const float* x1, float* out0, float* out1) {
    const float* xs[2] = {x0, x1};
    float* outs[2] = {out0, out1};
    rows_n(true, W, K, r0, r1, 2, xs, outs);
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
    if (j == 0) n_rows_ = 0;   // (wait() emptied the list)
    const int n_in = std::max(1, job->n_in);
    if (j >= kMaxJobs || n_rows_ + n_in > kMaxJobs || n_in > ExpertJob::kMaxIn) std::abort();
    if (F > scratch_F_) {   // (the first expert, or a wider model: no batch is in flight unless j > 0)
        if (j > 0) std::abort();
        // (every g / u / h starts a cache line - h is down's input, 32 bytes a load: F up to a multiple of 16,
        //  and 16 values of slack for the first line)
        scratch_F_ = (F + 15) / 16 * 16;
        scratch_.assign((size_t) kMaxJobs * 3 * scratch_F_ + 16, 0.0f);
    }
    job->C = C, job->F = F, job->gcap = gcap, job->ucap = ucap;
    job->n_in = n_in;
    // (input k's gate / up sums and activation: scratch row n_rows_ + k, as g / u / h + k * 3 * scratch_F_)
    float* const scratch = (float*) (((uintptr_t) scratch_.data() + 63) & ~(uintptr_t) 63);
    job->g = scratch + (size_t) n_rows_ * 3 * scratch_F_;
    n_rows_ += n_in;
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
// once all of an expert's activation is there - down's rows.  A row block's bytes are read once for all the
// inputs of its job (rows_n); a job of one input goes the same way.
void Pool::work() {
    for (;;) {
        const int n = n_jobs_.load(std::memory_order_acquire);
        bool took = false, pending = false;
        for (int j = 0; j < n; ++j) {
            ExpertJob& J = *jobs_[j];
            const int ni = J.n_in;
            const size_t ld = (size_t) 3 * scratch_F_;   // (an input's g / u / h)
            const float* const* xs = ni > 1 ? J.xs : &J.x;
            float* const* ys = ni > 1 ? J.ys : &J.y;
            const float* hs[ExpertJob::kMaxIn];
            float* o[ExpertJob::kMaxIn];
            while (J.next_a.load(std::memory_order_relaxed) < J.n_a) {
                const int i = J.next_a.fetch_add(1);
                if (i >= J.n_a) break;
                const int r0 = i * kRowsA, r1 = std::min(J.F, r0 + kRowsA);
                for (int q = 0; q < ni; ++q) o[q] = J.g + q * ld + r0;
                rows_n(J.q8, J.gate, J.C, r0, r1, ni, xs, o);
                for (int q = 0; q < ni; ++q) o[q] = J.u + q * ld + r0;
                rows_n(J.q8, J.up, J.C, r0, r1, ni, xs, o);
                for (int q = 0; q < ni; ++q)
                    activation(J.g + q * ld + r0, J.u + q * ld + r0, J.h + q * ld + r0, r1 - r0, J.sg, J.su, J.gcap,
                               J.ucap);
                J.done_a.fetch_add(1, std::memory_order_release);
                took = true;
            }
            if (J.done_a.load(std::memory_order_acquire) < J.n_a) {   // (its last blocks are still another thread's)
                pending = true;
                continue;
            }
            for (int q = 0; q < ni; ++q) hs[q] = J.h + q * ld;
            while (J.next_b.load(std::memory_order_relaxed) < J.n_b) {
                const int i = J.next_b.fetch_add(1);
                if (i >= J.n_b) break;
                const int r0 = i * kRowsB, r1 = std::min(J.C, r0 + kRowsB);
                for (int q = 0; q < ni; ++q) o[q] = ys[q] + r0;
                rows_n(J.q8, J.down, J.F, r0, r1, ni, hs, o);
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
