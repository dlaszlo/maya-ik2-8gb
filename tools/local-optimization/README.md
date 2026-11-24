# 128K cache optimization experiment

Base: `62688c7` on `avx2-cpu-lane`. The local original repository is
`/home/dlaszlo/maya-inkling-mini`. This checkout is an independent copy.

`IK2_CACHE_POLICY=schedule` adds a decode-only admission and eviction policy.
It estimates each expert's selection probability using an exponential moving
average (0.05 per visit of that layer), divided by the number of layer visits
until its layer is next used. The schedule comes from the loaded model, not
hardcoded Mini layer numbers. It retains likely, soon-needed experts and skips
background uploads that would replace a more valuable resident expert.

The default remains the existing `fifo` policy. Prefill still uses the existing
queues and scan-resistant policy. Current picks and in-flight uploads cannot
be evicted by the new policy; borrowed prefill scratch slots are excluded.
Cache capacity, KV representation, the 131072 context limit, expert count,
model weights, and routing are not changed by this option. CPU versus GPU
execution can still cause the existing floating-point differences.

The policy works alongside `IK2_X_CACHE_BIAS`. That separate option **does**
change routing and model output; combined quality/speed needs GPU measurement.
No claim is made that the improvements from the two options add together.
The new policy requires the single-GPU CPU lane with overlap and MV enabled;
it rejects incompatible profiling/CPU-disabled configurations.

## Verification performed

- Complete CUDA 13.0 / GCC 13 / sm_89 Release build: passed. No dependency
  downloads needed (`STRATA_NATIVE_EXPERTS=OFF` disables unrelated IQ engines).
- Replay of the saved 1199-token routing trace through the actual `ExpertCache`
  class extracted from `ik2_main.cu`, with CUDA copy operations stubbed:

  | Replay model | FIFO CPU experts/token | Schedule CPU experts/token | FIFO copies/token | Schedule copies/token |
  |---|---:|---:|---:|---:|
  | Upload ready at next visit | 109.58 | 98.68 | 37.21 | 30.35 |
  | Two extra visits of upload delay | 113.82 | 103.08 | 27.22 | 25.37 |

  Both halves of this trace improve. This is one coding prompt, not independent
  task coverage. Policy execution cost in this replay was approximately 0.1 ms
  per token. Queue/location consistency is asserted after every layer visit;
  current-pick/in-flight protection and `lend`/`give_back` are checked too.
- No live GPU inference has been run from the Codex sandbox: it has no
  `/dev/nvidia*` devices. The host's GPU is available according to the owner's
  `nvidia-smi`. These results are **not measured token/s improvements**.

Raw build and replay results: `validation/`.

## Run on the GPU host

Rebuild if source changes:

```bash
bash tools/local-optimization/build.sh
```

Compare original, FIFO, schedule, and the two policies with routing bias 0.1:

```bash
~/local-ai/maya-venv/bin/python tools/local-optimization/benchmark.py --biases 0,0.1
```

The benchmark reads the owner's current local 8GB configuration and requires
131072 context. It refuses to start if there is an active GPU compute process;
it never stops a service. It tests a **315-token prompt with 128K allocated**,
using 300 fixed continuation tokens to compare the same workload. It first
compares 127 next-token logit rows to the original engine (KL and top-token
agreement), then runs two timing rounds without logits I/O, reversing the
order on the second round. Logits are deleted after comparison; logs, exact
results and medians remain in a fresh `bench-128k-*/manifest.json` directory.

Unrelated environment variables, including GPU driver tweaks such as
`LD_PRELOAD`, are inherited. `IK2_*` variables are explicitly set per child
for a controlled comparison. It does not change the installed server config.

The default bias sweep also includes 0.2. For more repeats use `--repeats 3`.
To check near-full context, add `--prompt-length 130000`: this uses an artificial
repeated-token prefix and can take a long time to prefill. It tests capacity,
not long-document answer quality. A task-level quality evaluation is still
required before selecting an approximate routing configuration for daily use.
