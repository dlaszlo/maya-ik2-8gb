// Decode-only cache priorities. Learns from past/current routing, never future picks.
// (measurement copy, bench-2026-11-24/astra: mode and alpha are settable - IK2_SCHED_MODE, IK2_SCHED_ALPHA)
//   mode 0: the original - pick probability / visits until the layer's next visit
//   mode 1: pick probability x the layer's visits a token (how often the slot is used; no timetable)
//   mode 2: 1 / expected visits until the expert's next use = 1 / (d + period x (1 - p) / p)
#pragma once
#include <algorithm>
#include <vector>

namespace ik2::cache {
class Schedule {
public:
    bool enabled = false, active = false;
    int mode = 0;
    float alpha = 0.05f;
    bool admit_always = false;   // IK2_SCHED_ADMIT=1: no admission test (every allowed lazy copy is made)
    void init(const std::vector<int>& visits, int layers, int experts, int picks) {
        experts_ = experts;
        layers_ = layers;
        visits_ = (int) visits.size();
        probability_.assign(layers * experts, float(picks) / experts);
        inverse_distance_.assign(visits_ * layers, 0.0f);
        for (int v = 0; v < visits_; ++v)
            for (int d = 1; d <= visits_; ++d) {
                float& r = inverse_distance_[v * layers + visits[(v + d) % visits_]];
                if (r == 0.0f) r = 1.0f / d;
            }
        std::vector<int> n(layers, 0);
        for (int l : visits) ++n[l];
        period_.assign(layers, (float) visits_);
        for (int l = 0; l < layers; ++l)
            if (n[l]) period_[l] = (float) visits_ / n[l];
    }
    void observe(int visit, int layer, const int* picks, int count) {
        active = enabled;
        if (!active) return;
        visit_ = visit % visits_;
        // EWMA per visit of this layer, rather than per token: loop passes are observations too.
        const int start = layer * experts_;
        for (int e = 0; e < experts_; ++e) probability_[start + e] *= 1.0f - alpha;
        for (int i = 0; i < count; ++i) probability_[start + picks[i]] += alpha;
    }
    float value(int expert_index) const {
        const int l = expert_index / experts_;
        const float p = probability_[expert_index];
        if (mode == 1) return p / period_[l];
        const float inv_d = inverse_distance_[visit_ * layers_ + l];
        if (mode == 2) return 1.0f / (1.0f / inv_d + period_[l] * (1.0f - std::min(p, 1.0f)) / std::max(p, 1e-6f));
        return p * inv_d;
    }
private:
    int experts_ = 0, layers_ = 0, visits_ = 0, visit_ = 0;
    std::vector<float> probability_, inverse_distance_, period_;
};
}  // namespace ik2::cache
