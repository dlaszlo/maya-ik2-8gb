#!/usr/bin/env python3
"""Compare original/FIFO/schedule at 128K capacity, including cache-aware routing.
Run with local-ai/maya-venv/bin/python on the GPU host. Does not stop/start any service.
"""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
LOCAL = Path('/home/dlaszlo/local-ai')

def parse_log(text):
    m = re.search(r'^generation: (\d+) tokens in ([\d.]+) s \(([\d.]+) tok/s\)', text, re.M)
    if not m:
        raise RuntimeError('No generation timing in log')
    return {'tokens': int(m[1]), 'seconds': float(m[2]), 'tok_s': float(m[3])}

def compare_logits(ref, other, vocab=200058):
    import numpy as np
    a = np.memmap(ref, dtype=np.float32, mode='r')
    b = np.memmap(other, dtype=np.float32, mode='r')
    if a.size != b.size or a.size % vocab:
        raise RuntimeError('Incomplete or mismatched logits')
    a = a.reshape(-1, vocab); b = b.reshape(-1, vocab)
    kls = []; same = 0
    for i in range(0, len(a), 4):
        x = a[i:i+4].astype(np.float64); y = b[i:i+4].astype(np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise RuntimeError('Non-finite logits')
        same += int((x.argmax(1) == y.argmax(1)).sum())
        x -= x.max(axis=1, keepdims=True); y -= y.max(axis=1, keepdims=True)
        x -= np.log(np.exp(x).sum(axis=1, keepdims=True))
        y -= np.log(np.exp(y).sum(axis=1, keepdims=True))
        kls.extend((np.exp(x) * (x-y)).sum(axis=1).tolist())
    return {'rows': len(a), 'same_top': same, 'kl_mean': statistics.mean(kls), 'kl_max': max(kls)}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--biases', default='0,0.1,0.2')
    ap.add_argument('--quality-tokens', type=int, default=128)
    ap.add_argument('--prompt-length', type=int, default=0,
                    help='Artificially repeat the saved prompt to this length; 130000 tests near-full 128K (slow).')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if a.repeats < 1 or not 2 <= a.quality_tokens <= 300 or not 0 <= a.prompt_length <= 130000:
        ap.error('repeats >= 1; quality-tokens 2..300; prompt-length 0..130000')
    biases = [float(s) for s in a.biases.split(',')]
    if any(not 0 <= b <= 1 for b in biases): ap.error('biases must be in 0..1')
    cfg = json.loads((LOCAL/'maya/ik2-mini-8gb-local.json').read_text())
    binary = ROOT/'build-opt/maya-ik2'
    original = Path(cfg['exe'])
    if not a.dry_run:
        if not binary.is_file(): raise RuntimeError('Run build.sh first')
        subprocess.run(['nvidia-smi'], check=True)
        busy = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
        if busy: raise RuntimeError('GPU has a compute process; stop it before benchmarking: '+busy)
    out = Path(tempfile.mkdtemp(prefix='bench-128k-', dir=ROOT/'tools/local-optimization'))
    prompt = (LOCAL/'maya/bench-2026-11-24/gpu/game.ids').read_text().strip()
    ids = [int(s) for s in prompt.replace('\n', ',').split(',') if s.strip()]
    if a.prompt_length:
        ids = (ids*((a.prompt_length+len(ids)-1)//len(ids)))[:a.prompt_length]
    (out/'prompt.ids').write_text(','.join(map(str, ids)))
    old = (LOCAL/'maya/bench-2026-11-24/gpu/old128.log').read_text()
    force = re.search(r'^generated: (.+)$', old, re.M)[1].split()
    if len(force) != 300: raise RuntimeError('Expected 300 saved reference tokens')
    (out/'force.ids').write_text(','.join(force))
    # Clean IK2 experiment inheritance for each child; unrelated GPU tweaks (e.g. LD_PRELOAD) stay inherited.
    env0 = {k:v for k,v in os.environ.items() if not k.startswith('IK2_')}
    env0.update(cfg['env'])
    for k in list(env0):
        if k.startswith('IK2_X_'): del env0[k]
    env0.update(IK2_KV='fp8', IK2_CPU_SUM='1', IK2_STATS='1', IK2_FORCE=str(out/'force.ids'))
    cases = [('original', original, 'fifo', 0.0)]
    for b in biases:
        for policy in ['fifo','schedule']:
            cases.append((f'{policy}-bias-{b:g}', binary, policy, b))
    args = list(cfg['args'])
    if args[args.index('--max-context')+1] != '131072':
        raise RuntimeError('This benchmark requires configured 131072 context')
    manifest = {'context_capacity':131072,'prompt_tokens':len(ids),'forced_tokens':300,
                'repeats':a.repeats,'cases':[c[0] for c in cases], 'quality':{}, 'runs':[],
                'limitations':'One saved coding prompt; no task-level quality verdict. Fixed routing traces do not model GPU time.'}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Results:', out, flush=True)
    if a.dry_run:
        print(json.dumps(manifest, indent=2)); return
    def run(case, phase, gen, logits=None):
        name, exe, policy, bias = case
        env = env0.copy(); env.update(IK2_CACHE_POLICY=policy, IK2_X_CACHE_BIAS=str(bias))
        if logits: env['IK2_LOGITS_OUT'] = str(logits)
        cmd = [str(exe),*args,'--tokens','@'+str(out/'prompt.ids'),'--gen',str(gen)]
        path = out/f'{phase}-{name}.log'
        print(f'{phase}: {name}', flush=True)
        with path.open('w') as f:
            p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        if p.returncode: raise RuntimeError(f'Exit {p.returncode}: {path}')
        return parse_log(path.read_text())
    ref = out/'original.f32'
    for i,case in enumerate(cases):
        logits = ref if i==0 else out/'candidate.f32'
        run(case,'quality',a.quality_tokens,logits)
        manifest['quality'][case[0]] = compare_logits(ref, logits)
        print(json.dumps(manifest['quality'][case[0]]), flush=True)
        if i: logits.unlink()
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    ref.unlink()
    for rep in range(a.repeats):
        # Reverse alternating rounds so policy/bias ordering is not always the same.
        order = cases if rep%2==0 else cases[::-1]
        for case in order:
            result = run(case,f'speed-{rep+1}',300)
            result.update(case=case[0],repeat=rep+1)
            manifest['runs'].append(result)
            print(json.dumps(result), flush=True)
            (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    manifest['median_tok_s'] = {case[0]:statistics.median(r['tok_s'] for r in manifest['runs'] if r['case']==case[0]) for case in cases}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'median_tok_s':manifest['median_tok_s'],'quality':manifest['quality']}, indent=2),flush=True)

if __name__ == '__main__':
    main()
