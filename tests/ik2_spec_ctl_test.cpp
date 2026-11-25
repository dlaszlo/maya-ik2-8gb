// tests/ik2_spec_ctl_test.cpp - the drafting controller and its modes (src/ik2/ik2_spec_ctl.hpp), without a GPU.
//
//   g++ -O2 -std=c++20 -I src/ik2 tests/ik2_spec_ctl_test.cpp -o /tmp/ik2_spec_ctl_test && /tmp/ik2_spec_ctl_test
//
// 1. ModeTrack: the modes Inkling's markers give (thinking, text, each tool, an open code fence), a prompt's end,
//    more modes than slots.
// 2. SpecCtl on simulated text, with the cache path's measured times (RTX 4060 8 GB: a plain token 24.4 ms, a
//    window of b inputs and its drafting 45.2 .. 69.3 ms) and a draft accepted with a probability that is the
//    text's (prose 0.45, code 0.82 - the 1.87 and 3.50 tokens a 5-wide step measured): prose alone, code alone, and
//    an agent's turns (a short thinking, then a file written by a tool).  Against the controller before - one
//    statistic, a look every 8 plain tokens - and against plain decoding.
#include "ik2_spec_ctl.hpp"

#include <cstdio>
#include <random>

using ik2::ModeTrack;
using ik2::SpecCtl;

namespace {

int g_fail = 0;
#define CHECK(x)                                                         \
    do {                                                                 \
        if (!(x)) std::printf("FAIL line %d: %s\n", __LINE__, #x), ++g_fail; \
    } while (0)

// Inkling 2's ids (the tokenizer's): <|message_model|>, the content markers, <|end_message|>, the ``` tokens
constexpr int kModel = 200001, kThink = 200008, kText = 200004, kTool = 200049, kXml = 200024, kEnd = 200010;
constexpr int kFence = 68790, kFence2 = 158841;
ModeTrack inkling() { return ModeTrack({kModel}, {kThink, kText, kTool, kXml}, {kFence, kFence2}); }
// ... with the fixed switch: drafting in a tool call's content (and inside a fence), nowhere else
ModeTrack inkling_fixed() { return ModeTrack({kModel}, {kThink, kText, kTool, kXml}, {kFence, kFence2}, {kTool}); }

void feed(ModeTrack& t, std::initializer_list<int> ids) {
    for (int x : ids) t.see(x);
}

void test_modes() {
    ModeTrack off({}, {}, {});
    feed(off, {kModel, kThink, 5, 6});
    CHECK(off.mode() == 0);   // (no ids given: one mode)

    ModeTrack t = inkling();
    CHECK(t.mode() == 0);
    feed(t, {kModel, kThink, 11, 12});
    const int think = t.mode();
    CHECK(think > 0);
    feed(t, {kEnd, kModel});
    CHECK(t.mode() == 0);   // (a message's header: the tool's name is still coming)
    feed(t, {kText, 20, 21});
    const int text = t.mode();
    CHECK(text > 0 && text != think);
    feed(t, {kFence, 30, 31});
    const int text_code = t.mode();
    CHECK(text_code != text && text_code != think);
    feed(t, {kFence2});
    CHECK(t.mode() == text);   // (the fence closed)
    // a tool call: <|message_model|> name <|content_invoke_tool_json|> - each tool its own mode
    feed(t, {kEnd, kModel, 700, kTool, 1, 2});
    const int write = t.mode();
    feed(t, {kEnd, kModel, 701, 702, kTool, 1, 2});
    const int edit = t.mode();
    CHECK(write != edit && write != text && edit != think);
    feed(t, {kEnd, kModel, 700, kTool});
    CHECK(t.mode() == write);   // (the same tool again: the same mode)
    feed(t, {kFence});
    CHECK(t.mode() != write);
    feed(t, {kEnd, kModel, kThink});
    CHECK(t.mode() == think);   // (a new message: the fence is not open in it)

    // a prompt's end: the mode its last message is in
    ModeTrack p = inkling();
    p.begin({1, 2, kModel, kText, 3, kFence, 4, kFence, kEnd, 200000, 9, kModel, 700, kTool, 5});
    ModeTrack q = inkling();
    feed(q, {kModel, 700, kTool, 5});
    CHECK(p.base == q.base && !p.fence);
    p.begin({kModel, kText, 3, kFence, 4});
    CHECK(p.fence);
    p.begin({1, 2, 3});
    CHECK(p.mode() == 0);

    // more modes than slots: the last slot takes the rest, nothing runs over
    ModeTrack m = inkling();
    for (int i = 0; i < 100; ++i) {
        feed(m, {kEnd, kModel, 1000 + i, kTool, 1});
        CHECK(m.mode() >= 1 && m.mode() < ModeTrack::kModes);
    }
    CHECK((int) m.slot_toks.size() <= ModeTrack::kModes);
    feed(m, {kEnd, kModel, 1000, kTool, 1});
    CHECK(m.mode() == 1);   // (the first ones keep their slots)

    // the fixed switch: on in a tool call and inside a fence, off in the thinking, an answer's prose and a header
    ModeTrack l = inkling();
    CHECK(!l.fixed() && l.drafts());   // (no IK2_MODE_DRAFT: every mode may draft)
    ModeTrack f = inkling_fixed();
    CHECK(f.fixed() && !f.drafts());
    feed(f, {kModel, kThink, 1, 2});
    CHECK(!f.drafts());
    feed(f, {kEnd, kModel, 700});
    CHECK(!f.drafts());   // (the tool's name)
    feed(f, {kTool, 1, 2});
    CHECK(f.drafts());
    feed(f, {kEnd, kModel, kText, 5});
    CHECK(!f.drafts());
    feed(f, {kFence, 6});
    CHECK(f.drafts());
    feed(f, {kFence, 7});
    CHECK(!f.drafts());
    f.begin({kModel, 700, kTool, 5});
    CHECK(f.drafts());
}

// The controller before this change: one statistic, a look every `probe` plain tokens.
struct OldCtl {
    int wmax = 8, probe = 8, probe_w = 4, plain_run = 0;
    double c_plain = 0, c[ik2::kSpecBlockMax + 1] = {}, S = 1, F = 1, decay = 0.85;
    int choose(int room) {
        const int bmax = std::min(wmax, room);
        if (bmax < 2) return 1;
        const double a = std::min(0.98, (S + 0.5) / (S + F + 1.0));
        int best = 1;
        double vb = 1.0 / c_plain;
        for (int b = 2; b <= bmax; ++b) {
            const double e = (1.0 - std::pow(a, b)) / (1.0 - a), v = e / c[b];
            if (v > vb) vb = v, best = b;
        }
        if (best == 1 && plain_run >= probe) best = std::min(probe_w, bmax);
        return best;
    }
    void took_plain(double t) { c_plain = 0.9 * c_plain + 0.1 * t, ++plain_run; }
    void took_spec(int B, int acc, double t) {
        c[B] = 0.8 * c[B] + 0.2 * t;
        S = decay * S + acc, F = decay * F + (acc < B - 1 ? 1 : 0);
        plain_run = 0;
    }
};

// the cache path's times, s: a plain token; a window of b inputs with its drafting
const double kPlain = 24.4e-3;
const double kWin[ik2::kSpecBlockMax + 1] = {0, 0, 45.2e-3, 49.8e-3, 54.3e-3, 58.2e-3, 62.1e-3, 65.7e-3, 69.3e-3};

template <class C>
void priors(C& c) {
    c.wmax = 8, c.c_plain = kPlain;
    for (int b = 2; b <= ik2::kSpecBlockMax; ++b) c.c[b] = kWin[b];
}

struct Seg {
    std::vector<int> open;   // the tokens that open it (markers: they are generated too)
    int n;                   // its tokens
    double a;                // a draft's acceptance in it
};

// tokens a second: the segments in order, `rounds` times; the tracker (if any) sees every token
template <class C>
double run(C& ctl, ModeTrack* track, const std::vector<Seg>& segs, int rounds, uint64_t seed, int* mode_of = nullptr) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0, 1);
    double t = 0;
    long tokens = 0;
    for (int r = 0; r < rounds; ++r)
        for (size_t si = 0; si < segs.size(); ++si) {
            const Seg& s = segs[si];
            for (int x : s.open) {
                if (track) track->see(x);
                t += kPlain, ++tokens;   // (a marker: a plain token, no step's statistics)
            }
            for (int left = s.n; left > 0;) {
                int B = 1;
                if constexpr (requires { ctl.mode; }) {
                    if (track) ctl.mode = track->mode(), ctl.allow = track->drafts(), ctl.eager = track->fixed();
                    if (mode_of) mode_of[si] = ctl.mode;
                }
                B = ctl.choose(left);
                if (B < 2) {
                    ctl.took_plain(kPlain);
                    t += kPlain, ++tokens, --left;
                    if (track) track->see(7);
                    continue;
                }
                int acc = 0;
                while (acc < B - 1 && u(rng) < s.a) ++acc;
                ctl.took_spec(B, acc, kWin[B]);
                const int got = std::min(left, acc + 1);
                t += kWin[B], tokens += got, left -= got;
                for (int j = 0; j < got; ++j)
                    if (track) track->see(7);
            }
        }
    return tokens / t;
}

void test_controller() {
    const double plain = 1.0 / kPlain;
    const Seg prose{{}, 4000, 0.45}, code{{}, 4000, 0.82};
    // an agent's turn: a short thinking, a file written by a tool, a short answer
    const std::vector<Seg> turn = {{{kModel, kThink}, 60, 0.45},
                                   {{kEnd, kModel, 700, kTool}, 900, 0.82},
                                   {{kEnd, kModel, kText}, 80, 0.45}};
    // ... and one where the file is prose (a README): the switch says draft, the acceptance has to step back
    const std::vector<Seg> turn_prose = {{{kModel, kThink}, 60, 0.45}, {{kEnd, kModel, 700, kTool}, 900, 0.45}};
    double so[3] = {}, sn[3] = {}, sf[3] = {};
    const int kSeeds = 20;
    for (int seed = 1; seed <= kSeeds; ++seed) {
        {
            OldCtl o;
            SpecCtl n;
            priors(o), priors(n);
            so[0] += run(o, nullptr, {prose}, 1, seed), sn[0] += run(n, nullptr, {prose}, 1, seed);
        }
        {
            OldCtl o;
            SpecCtl n;
            priors(o), priors(n);
            so[1] += run(o, nullptr, {code}, 1, seed), sn[1] += run(n, nullptr, {code}, 1, seed);
        }
        {
            OldCtl o;
            SpecCtl n;
            priors(o), priors(n);
            ModeTrack tr = inkling();
            so[2] += run(o, nullptr, turn, 8, seed), sn[2] += run(n, &tr, turn, 8, seed);
        }
        {   // the fixed switch: prose in an answer's text, the turns, the turns with a file of prose
            const std::vector<Seg> answer = {{{kModel, kText}, 4000, 0.45}};
            SpecCtl a, b, c2;
            priors(a), priors(b), priors(c2);
            ModeTrack ta = inkling_fixed(), tb = inkling_fixed(), tc = inkling_fixed();
            sf[0] += run(a, &ta, answer, 1, seed), sf[1] += run(b, &tb, turn, 8, seed);
            sf[2] += run(c2, &tc, turn_prose, 8, seed);
            CHECK(a.steps == 0);   // (prose: not one window)
        }
    }
    const char* name[3] = {"prose alone", "code alone", "an agent's turns"};
    std::printf("tokens a second (a plain token %.1f):\n", plain);
    for (int i = 0; i < 3; ++i)
        std::printf("  %-18s before %.1f, now %.1f (%+.1f%%)\n", name[i], so[i] / kSeeds, sn[i] / kSeeds,
                    100.0 * (sn[i] / so[i] - 1.0));
    CHECK(sn[0] / kSeeds > 0.985 * plain);     // prose: within 1.5% of not drafting at all
    CHECK(sn[0] > so[0]);                      // ... and better than before
    CHECK(sn[1] > 0.99 * so[1]);               // code: nothing lost (within the simulation's noise)
    CHECK(sn[2] > so[2]);                      // turns: the modes pay
    std::printf("  the fixed switch:  prose %.1f, an agent's turns %.1f, turns with a file of prose %.1f\n",
                sf[0] / kSeeds, sf[1] / kSeeds, sf[2] / kSeeds);
    CHECK(std::fabs(sf[0] / kSeeds - plain) < 1e-6 * plain);   // prose: a plain token's speed exactly
    CHECK(sf[1] >= sn[2]);                                     // turns: at least what the learned modes give
    CHECK(sf[2] / kSeeds > 0.97 * plain);                      // a file of prose: the acceptance steps back
    // a mode's statistics are its own: after the turns, thinking is looked at rarely, the tool's mode drafts
    SpecCtl n;
    priors(n);
    ModeTrack tr = inkling();
    int mode_of[3] = {};
    run(n, &tr, turn, 8, 7, mode_of);
    const SpecCtl::Mode &think = n.modes[mode_of[0]], &tool = n.modes[mode_of[1]];
    CHECK(mode_of[0] != mode_of[1]);
    CHECK(tool.steps > 20 * std::max<uint64_t>(1, think.steps));
    CHECK(tool.plain < tool.steps);
    CHECK(think.probe_iv > n.probe);
    std::printf("  thinking: %llu windows, %llu plain, the next look after %d; the tool: %llu windows, %llu plain\n",
                (unsigned long long) think.steps, (unsigned long long) think.plain, think.probe_iv,
                (unsigned long long) tool.steps, (unsigned long long) tool.plain);
}

}  // namespace

int main() {
    test_modes();
    test_controller();
    std::printf(g_fail ? "FAILED (%d)\n" : "PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
