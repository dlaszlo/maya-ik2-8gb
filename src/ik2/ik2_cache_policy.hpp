// Decode-only cache priorities. Learns from past/current routing, never future picks.
#pragma once
#include <algorithm>
#include <vector>

namespace ik2::cache {
class Schedule {
public:
    bool enabled = false, active = false;
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
    }
    void observe(int visit, int layer, const int* picks, int count) {
        active = enabled;
        if (!active) return;
        visit_ = visit % visits_;
        // EWMA per visit of this layer, rather than per token: loop passes are observations too.
        constexpr float alpha = 0.05f;
        const int start = layer * experts_;
        for (int e = 0; e < experts_; ++e) probability_[start + e] *= 1.0f - alpha;
        for (int i = 0; i < count; ++i) probability_[start + picks[i]] += alpha;
    }
    float value(int expert_index) const {
        return probability_[expert_index] * inverse_distance_[visit_ * layers_ + expert_index / experts_];
    }
private:
    int experts_ = 0, layers_ = 0, visits_ = 0, visit_ = 0;
    std::vector<float> probability_, inverse_distance_;
};
}  // namespace ik2::cache
