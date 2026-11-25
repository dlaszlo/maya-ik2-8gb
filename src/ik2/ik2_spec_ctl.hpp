// src/ik2/ik2_spec_ctl.hpp - maya-ik2's drafting controller: when to draft and how wide, and the mode the model is
// writing in (host code only, so tests/ik2_spec_ctl_test.cpp can run it without a GPU).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace ik2 {

constexpr int kSpecBlockMax = 8;   // (ik2_main.cu's kBlockMax: the widest window)

// What the model is writing, read off its own tokens: the drafting's mode.  A message starts with a start token
// (IK2_MODE_START: Inkling's <|message_model|>), then - for a tool call - the tool's name, then a content marker
// (IK2_MODE_MARKS: <|content_thinking|>, <|content_text|>, <|content_invoke_tool_json|>, ...).  A mode is the tokens
// from the start to the marker - thinking, an answer's text, each tool its own - and whether a code fence is open in
// it (IK2_MODE_FENCE: the tokens with ``` in them toggle it).  The ids are the server's to give, it has the
// tokenizer; with none there is one mode.
// IK2_MODE_DRAFT (content markers) makes the switch a fixed one: drafting is on where the model writes one of these
// contents (Inkling's tool calls: a file's text, a command) and inside a code fence, off everywhere else (the
// thinking, an answer's prose) - no looks there, no draft's time.  The engine runs one family of models, so the rule
// can be the model's own format; without IK2_MODE_DRAFT every mode drafts when its own acceptance says so.
// A mode never changes what the model computes or the tokens chosen: only whether and how wide a window is tried.
struct ModeTrack {
    static constexpr int kModes = 32;   // slots; the last takes every mode after the first kModes - 1
    static constexpr int kHead = 16;    // a header's tokens that name the mode (a tool's name)
    static constexpr int kScan = 8192;  // a prompt's last tokens begin() reads
    std::vector<int> starts, marks, fences, draft_marks;
    int base_mark = -1;   // the marker the current content began with
    std::unordered_map<uint64_t, int> slot_of;
    std::vector<std::vector<int>> slot_toks;   // a slot's header tokens and marker (what the statistics print)
    std::vector<bool> slot_fence;
    std::vector<int> head;
    bool in_head = false, fence = false;
    uint64_t base = 0;
    std::vector<int> base_toks;

    static std::vector<int> ids_of(const char* env) {
        std::vector<int> v;
        const char* s = getenv(env);
        if (!s) return v;
        std::stringstream in(s);
        for (std::string x; std::getline(in, x, ',');)
            if (!x.empty()) v.push_back(atoi(x.c_str()));
        return v;
    }
    ModeTrack()
        : ModeTrack(ids_of("IK2_MODE_START"), ids_of("IK2_MODE_MARKS"), ids_of("IK2_MODE_FENCE"),
                    ids_of("IK2_MODE_DRAFT")) {}
    ModeTrack(std::vector<int> st, std::vector<int> mk, std::vector<int> fc, std::vector<int> dm = {})
        : starts(std::move(st)), marks(std::move(mk)), fences(std::move(fc)), draft_marks(std::move(dm)) {
        slot_toks.push_back({});   // (slot 0: before any marker, and a message's header)
        slot_fence.push_back(false);
    }
    bool on() const { return !marks.empty(); }
    // the fixed switch (IK2_MODE_DRAFT), and whether drafting is on where the model is now
    bool fixed() const { return on() && !draft_marks.empty(); }
    bool drafts() const { return !fixed() || (base && (fence || has(draft_marks, base_mark))); }
    static bool has(const std::vector<int>& v, int t) { return std::find(v.begin(), v.end(), t) != v.end(); }
    void reset() {
        head.clear(), base_toks.clear();
        in_head = fence = false, base = 0, base_mark = -1;
    }
    // a token the model chose (or a prompt's), in order
    void see(int t) {
        if (!on()) return;
        if (has(starts, t)) {   // a new message: its header until the marker
            reset();
            in_head = true;
            return;
        }
        if (has(marks, t)) {
            uint64_t h = 1469598103934665603ull;   // (FNV-1a over the header's tokens and the marker)
            base_toks = head;
            base_toks.push_back(t);
            for (int x : base_toks) h = (h ^ (uint64_t) (uint32_t) x) * 1099511628211ull;
            base = h ? h : 1, in_head = fence = false, base_mark = t;
            head.clear();
            return;
        }
        if (in_head) {
            if ((int) head.size() < kHead) head.push_back(t);
            return;
        }
        if (base && has(fences, t)) fence = !fence;
    }
    // a request's prompt: the mode its end is in (its last kScan tokens, read in order)
    void begin(const std::vector<int>& ids) {
        reset();
        for (size_t i = ids.size() > (size_t) kScan ? ids.size() - kScan : 0; i < ids.size(); ++i) see(ids[i]);
    }
    int mode() {
        if (!on() || !base) return 0;
        const uint64_t key = base * 2 + (fence ? 1 : 0);
        const auto it = slot_of.find(key);
        if (it != slot_of.end()) return it->second;
        const int s = std::min((int) slot_toks.size(), kModes - 1);
        if (s == (int) slot_toks.size()) slot_toks.push_back(base_toks), slot_fence.push_back(fence);
        slot_of.emplace(key, s);
        return s;
    }
};

// When to draft and how wide (IK2_DRAFT_WIDTH the widest; IK2_DRAFT_ADAPT=0: always that wide): a step's expected
// tokens (1 - a^B) / (1 - a) - a the drafts' acceptance, each accepted given the one before it, from the recent
// steps - over its measured time, against a plain token's; a plain run drafts once in a while to look again.
// The acceptance and the looks are a mode's own (ModeTrack: `mode`, set before choose()): code written into a file
// accepts three drafts in four, prose one in three, and one statistic for both drafts where it loses and waits where
// it would win.  A mode that keeps losing is looked at less and less often: after IK2_DRAFT_PROBE plain tokens (8),
// twice that after a look with no draft accepted (up to IK2_DRAFT_PROBE_MAX, 128), half after one with every draft
// accepted, and IK2_DRAFT_PROBE again once drafting wins in it; a mode seen for the first time is looked at at once.
// The windows' and a plain token's times are the same in every mode.
// With the fixed switch (ModeTrack::fixed): `allow` false is a plain token, no look; a mode that drafts starts
// drafting at once (`eager`: its first statistics say it pays) and the acceptance only picks the width from there -
// and steps back when a file's text turns out to be prose.
struct SpecCtl {
    int wmax = 5;
    bool adapt = true;
    double c_plain = 7.8e-3, c[kSpecBlockMax + 1] = {};
    int probe = getenv("IK2_DRAFT_PROBE") ? atoi(getenv("IK2_DRAFT_PROBE")) : 8;   // plain tokens before a look
    int probe_max = getenv("IK2_DRAFT_PROBE_MAX") ? atoi(getenv("IK2_DRAFT_PROBE_MAX")) : 128;
    int probe_w = 4;
    double decay = getenv("IK2_DRAFT_DECAY") ? atof(getenv("IK2_DRAFT_DECAY")) : 0.85;
    struct Mode {
        double S = 1, F = 1;   // (the drafts accepted, the ones rejected: decayed counts)
        int plain_run = -1;    // (-1: not seen yet)
        int probe_iv = 0;      // plain tokens before this mode's next look
        uint64_t steps = 0, plain = 0, drafted = 0, accepted = 0, looks = 0;
    };
    Mode modes[ModeTrack::kModes];
    int mode = 0;
    bool allow = true, eager = false;   // (the mode's: set with it, before choose())
    bool look = false;   // (the step choose() returned is a look)
    uint64_t steps = 0, plain = 0, drafted = 0, accepted = 0;
    uint64_t hist[kSpecBlockMax] = {};
    SpecCtl() {
        const double prior[kSpecBlockMax + 1] = {0,       7.8e-3,  13.3e-3, 15.1e-3, 16.7e-3,
                                                 18.2e-3, 19.8e-3, 21.3e-3, 22.8e-3};
        for (int b = 0; b <= kSpecBlockMax; ++b) c[b] = prior[b];
        probe = std::max(1, probe), probe_max = std::max(probe, probe_max);
    }
    Mode& cur() {
        Mode& m = modes[std::clamp(mode, 0, ModeTrack::kModes - 1)];
        if (m.plain_run < 0) {   // (a new mode: a look at once - or, where the fixed switch drafts, windows at once)
            m.plain_run = probe, m.probe_iv = probe;
            if (eager && allow) m.S = 3;
        }
        return m;
    }
    int choose(int room) {   // room: the widest a step may be (1: plain)
        const int bmax = std::min(wmax, room);
        if (bmax < 2 || !allow) return 1;
        if (!adapt) return bmax;
        Mode& m = cur();
        const double a = std::min(0.98, (m.S + 0.5) / (m.S + m.F + 1.0));
        int best = 1;
        double vb = 1.0 / c_plain;
        for (int b = 2; b <= bmax; ++b) {
            const double e = (1.0 - std::pow(a, b)) / (1.0 - a), v = e / c[b];
            if (v > vb) vb = v, best = b;
        }
        if (best > 1) {
            m.probe_iv = probe;   // (drafting wins here: the next plain run is looked at soon again)
            return best;
        }
        if (m.plain_run >= m.probe_iv) {   // (a look at the acceptance again)
            ++m.looks;
            look = true;
            return std::min(probe_w, bmax);
        }
        return 1;
    }
    void took_plain(double t) {
        Mode& m = cur();
        c_plain = 0.9 * c_plain + 0.1 * t;
        ++plain, ++m.plain, ++m.plain_run;
    }
    void took_spec(int B, int acc, double t) {
        Mode& m = cur();
        c[B] = 0.8 * c[B] + 0.2 * t;
        m.S = decay * m.S + acc, m.F = decay * m.F + (acc < B - 1 ? 1 : 0);
        ++steps, drafted += B - 1, accepted += acc, ++hist[acc];
        ++m.steps, m.drafted += B - 1, m.accepted += acc;
        m.plain_run = 0;
        if (look) {   // (the next look: later after one with nothing accepted, sooner after one with everything)
            if (acc == 0) m.probe_iv = std::min(probe_max, 2 * m.probe_iv);
            else if (acc == B - 1) m.probe_iv = std::max(probe, m.probe_iv / 2);
            look = false;
        }
    }
};

}  // namespace ik2
