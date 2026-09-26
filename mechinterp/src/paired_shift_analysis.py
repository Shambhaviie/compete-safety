"""
paired_shift_analysis.py

Per-prompt paired A->C activation shift analysis.

For every prompt i (all 676):
  delta_i = h_i_C - h_i_A  (paired difference at each layer)
  score_i = delta_i . r_hat_refusal  (projection onto within-C refusal direction)

Negative score = competition pushed activations AWAY from refusal direction.
Positive score = competition pushed activations TOWARD refusal direction.

Compare distribution of scores for:
  - 163 flipped prompts (A=refused, C=complied)
  - 513 stable prompts  (A=refused, C=refused)

This is clean: uses the within-C refusal direction (computed from different data)
applied to the paired shift. No circularity.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

ACT_DIR   = Path('mechinterp/activations')
PROBE_DIR = Path('mechinterp/probes')
OUT_DIR   = Path('mechinterp/results')

print('='*65)
print('PAIRED A->C SHIFT ANALYSIS')
print('Per-prompt: how much does competition shift activations')
print('along the anti-refusal axis?')
print('='*65)

# Load activations
data_A = np.load(str(ACT_DIR / 'activations_prehook_A_seed42.npz'))
data_C = np.load(str(ACT_DIR / 'activations_prehook_C_seed42.npz'))
acts_A = data_A['prefill_acts'].astype(np.float32)
acts_C = data_C['prefill_acts'].astype(np.float32)
rl_A   = data_A['replayed_labels'].astype(bool)
rl_C   = data_C['replayed_labels'].astype(bool)
meta_A = json.load(open(ACT_DIR / 'metadata_prehook_A_seed42.json'))
meta_C = json.load(open(ACT_DIR / 'metadata_prehook_C_seed42.json'))

# Within-C refusal directions (built from C_train refused vs complied)
# These are computed from DIFFERENT data than the A vs C comparison
within_C_dirs = np.load(str(PROBE_DIR / 'directions_prehook_all_layers.npy')).astype(np.float32)

pid_to_idx_A = {m['prompt_id']: i for i, m in enumerate(meta_A)}
pid_to_idx_C = {m['prompt_id']: i for i, m in enumerate(meta_C)}
common_pids  = sorted(set(pid_to_idx_A.keys()) & set(pid_to_idx_C.keys()))

flipped_pids = []
stable_pids  = []
for pid in common_pids:
    a_refused = rl_A[pid_to_idx_A[pid]]
    c_refused  = rl_C[pid_to_idx_C[pid]]
    if a_refused and not c_refused:
        flipped_pids.append(pid)
    elif a_refused and c_refused:
        stable_pids.append(pid)

print(f'Flipped (A=refused, C=complied): {len(flipped_pids)}')
print(f'Stable  (A=refused, C=refused):  {len(stable_pids)}')
print(f'Using within-C refusal directions (no circularity)')

TARGET_LAYERS = [7, 8, 9, 10, 11, 14, 18, 24, 31]

print(f'\n{"Layer":<8} {"Flipped shift mean":<22} {"Stable shift mean":<22} {"Cohen d":<10} {"Mann-Whitney p"}')
print('-'*80)

results = {}
for L in TARGET_LAYERS:
    d = within_C_dirs[L]
    norm = np.linalg.norm(d)
    if norm < 1e-6:
        continue
    d = d / norm

    # Per-prompt paired shift projected onto refusal direction
    # Negative = moved away from refusal (toward compliance)
    flip_scores = np.array([
        (acts_C[pid_to_idx_C[p], L, :] - acts_A[pid_to_idx_A[p], L, :]) @ d
        for p in flipped_pids
    ])
    stab_scores = np.array([
        (acts_C[pid_to_idx_C[p], L, :] - acts_A[pid_to_idx_A[p], L, :]) @ d
        for p in stable_pids
    ])

    # Cohen's d effect size
    pooled_std = np.sqrt((flip_scores.std()**2 + stab_scores.std()**2) / 2)
    cohens_d   = (flip_scores.mean() - stab_scores.mean()) / pooled_std if pooled_std > 0 else 0

    # Mann-Whitney U test (non-parametric, no normality assumption)
    stat, pval = stats.mannwhitneyu(flip_scores, stab_scores, alternative='less')

    results[L] = {
        'flip_mean': float(flip_scores.mean()),
        'flip_std':  float(flip_scores.std()),
        'stab_mean': float(stab_scores.mean()),
        'stab_std':  float(stab_scores.std()),
        'cohens_d':  float(cohens_d),
        'mw_pval':   float(pval),
    }

    print(f'{L:<8} '
          f'{flip_scores.mean():>+8.3f} ± {flip_scores.std():.3f}      '
          f'{stab_scores.mean():>+8.3f} ± {stab_scores.std():.3f}      '
          f'{cohens_d:>+8.3f}   '
          f'{pval:.2e}')

print()
print('Interpretation:')
print('  Negative shift mean = competition moved activations AWAY from refusal direction')
print('  Flipped prompts should show more negative shift than stable prompts')
print('  Cohen d < 0 means flipped prompts shifted more negatively (expected)')
print('  p < 0.05 means the difference between flip and stable distributions is significant')

print()
print('Full distribution summary at layer 18:')
L = 18
d = within_C_dirs[L] / np.linalg.norm(within_C_dirs[L])
flip_scores = np.array([
    (acts_C[pid_to_idx_C[p], L, :] - acts_A[pid_to_idx_A[p], L, :]) @ d
    for p in flipped_pids
])
stab_scores = np.array([
    (acts_C[pid_to_idx_C[p], L, :] - acts_A[pid_to_idx_A[p], L, :]) @ d
    for p in stable_pids
])
print(f'  Flipped (n={len(flip_scores)}):  min={flip_scores.min():.3f}  '
      f'Q1={np.percentile(flip_scores,25):.3f}  '
      f'median={np.median(flip_scores):.3f}  '
      f'Q3={np.percentile(flip_scores,75):.3f}  '
      f'max={flip_scores.max():.3f}')
print(f'  Stable  (n={len(stab_scores)}): min={stab_scores.min():.3f}  '
      f'Q1={np.percentile(stab_scores,25):.3f}  '
      f'median={np.median(stab_scores):.3f}  '
      f'Q3={np.percentile(stab_scores,75):.3f}  '
      f'max={stab_scores.max():.3f}')

with open(OUT_DIR / 'paired_shift_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved: mechinterp/results/paired_shift_analysis.json')
