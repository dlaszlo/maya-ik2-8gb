// src/ik2/ik2_cpu.hpp - maya-ik2's CPU lane: a routed expert that is not in VRAM computed where it is, in host
// memory, while the GPU runs the experts it holds - instead of waiting for the expert's 18 MB over PCIe.
// NVFP4 and Q8_0 experts (AVX-512 F / BW / VL, or AVX2 + FMA + F16C; the weights read as they are stored), a few worker threads sharing an expert's
// rows.  The arithmetic is ref/inkling2.py's in F32 (the activation not rounded to FP16 as the device's is).
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace ik2::cpu {

/// Whether this CPU (and OS) runs the AVX-512 or the AVX2 kernels.
bool available();

struct ExpertJob;

/// Worker threads sharing the rows of the experts posted (post_expert), a few rows a time from one counter an expert,
/// so no thread is left with a share of its own to finish; wait(): until every expert posted so far is done.  At most kMaxJobs experts between two wait()s.
class Pool {
public:
    static constexpr int kMaxJobs = 64;
    explicit Pool(int n);
    ~Pool();
    void add(ExpertJob* job, int C, int F, float gcap, float ucap);
    void wait();
    int size() const { return (int) th_.size(); }

private:
    void work();
    std::vector<std::thread> th_;
    ExpertJob* jobs_[kMaxJobs] = {};
    std::atomic<int> n_jobs_{0}, left_{0}, active_{0};
    alignas(64) std::atomic<uint64_t> epoch_{0};   // (a line of its own: idle workers wait for a write to it)
    alignas(64) std::atomic<bool> stop_{false};
    std::vector<float> scratch_;   // gate's, up's sums and the activation of every job, [kMaxJobs][3][F]
    int scratch_F_ = 0;
    std::mutex m_;
    std::condition_variable cv_;
    int sleepers_ = 0;
};

/// out[r - r0] = W[r] . x for rows r0 .. r1 - 1 of an NVFP4 matrix (rows of K values, K % 64 == 0).
void nvfp4_rows(const uint8_t* W, int K, int r0, int r1, const float* x, float* out);
/// The same for a Q8_0 matrix (ggml's 34-byte blocks: an FP16 scale, then 32 int8 values; K % 32 == 0).
void q8_0_rows(const uint8_t* W, int K, int r0, int r1, const float* x, float* out);

/// One NVFP4 expert: gate, up [F][C] and down [C][F] blocks, x [C]; y [C] = down . h, h = gcap tanh(g / gcap)
/// sigmoid(g) ucap tanh(u / ucap), g = (gate . x) sg, u = (up . x) su - the raw sums of down (its scale is the
/// caller's).  The rows shared by the pool's threads; done when pool.wait() returns.  job must live until then.
struct ExpertJob {
    const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;
    const float* x = nullptr;
    float sg = 1.0f, su = 1.0f;
    bool q8 = false;   // the three matrices are Q8_0 instead of NVFP4
    float* y = nullptr;
    // (the pool's: the shapes, its scratch rows and the row blocks taken and done of gate / up and of down)
    int C = 0, F = 0, n_a = 0, n_b = 0;
    float gcap = 0.0f, ucap = 0.0f;
    float *g = nullptr, *u = nullptr, *h = nullptr;
    std::atomic<int> next_a{0}, done_a{0}, next_b{0}, done_b{0};
};
void post_expert(Pool& pool, ExpertJob* job, int C, int F, float gcap, float ucap);

}  // namespace ik2::cpu
