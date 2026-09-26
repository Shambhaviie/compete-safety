import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

ACT_DIR   = Path('mechinterp/activations')
PROBE_DIR = Path('mechinterp/probes')
OUT_DIR   = Path('mechinterp/results')
OUT_DIR.mkdir(parents=True, exist_ok=True)
N_LAYERS  = 32

print('=' * 65)
print('CONDITION A vs C — COMPETITION-INDUCED COMPLIANCE DIRECTION')
print('=' * 65)

data_A = np.load(str(ACT_DIR / 'activations_prehook_A_seed42.npz'))
data_C = np.load(str(ACT_DIR / 'activations_prehook_C_seed42.npz'))
acts_A = data_A['prefill_acts'].astype(np.float32)
acts_C = data_C['prefill_acts'].astype(np.float32)
rl_A   = data_A['replayed_labels'].astype(bool)
rl_C   = data_C['replayed_labels'].astype(bool)
meta_A = json.load(open(ACT_DIR / 'metadata_prehook_A_seed42.json'))
meta_C = json.load(open(ACT_DIR / 'metadata_prehook_C_seed42.json'))

print(f'Condition A: refused={rl_A.sum()} complied={(~rl_A).sum()}')
print(f'Condition C: refused={rl_C.sum()} complied={(~rl_C).sum()}')

pid_to_idx_A = {m['prompt_id']: i for i, m in enumerate(meta_A)}
pid_to_idx_C = {m['prompt_id']: i for i, m in enumerate(meta_C)}
common_pids  = set(pid_to_idx_A.keys()) & set(pid_to_idx_C.keys())
print(f'Common prompts: {len(common_pids)}')

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

within_C_dirs = np.load(str(PROBE_DIR / 'directions_prehook_all_layers.npy')).astype(np.float32)

print()
print(f'{"Layer":<8} {"Dir norm":<12} {"Cosine to within-C dir":<25}')
print('-'*48)

comp_directions = []
comp_norms      = []

for L in range(N_LAYERS):
    A_acts = np.array([acts_A[pid_to_idx_A[pid], L, :] for pid in flipped_pids])
    C_acts = np.array([acts_C[pid_to_idx_C[pid], L, :] for pid in flipped_pids])
    direction = C_acts.mean(0) - A_acts.mean(0)
    norm      = np.linalg.norm(direction)
    d_norm    = direction / norm if norm > 1e-6 else direction
    cosine    = float(np.dot(d_norm, within_C_dirs[L])) if norm > 1e-6 else 0.0
    comp_directions.append(d_norm)
    comp_norms.append(float(norm))
    if L in [0,4,7,8,9,10,11,12,15,16,17,18,19,24,31]:
        print(f'{L:<8} {norm:<12.4f} {cosine:<+.4f}')

comp_directions = np.array(comp_directions)

print()
print(f'{"Layer":<8} {"A_refused (flipped)":<28} {"C_complied (flipped)":<28} {"C_refused (stable)"}')
print('-'*85)

for L in [7,8,9,10,11,18]:
    d = comp_directions[L]
    if comp_norms[L] < 1e-6: continue
    A_proj = np.array([acts_A[pid_to_idx_A[p],L,:] @ d for p in flipped_pids])
    C_comp = np.array([acts_C[pid_to_idx_C[p],L,:] @ d for p in flipped_pids])
    C_stab = np.array([acts_C[pid_to_idx_C[p],L,:] @ d for p in stable_pids]) if stable_pids else np.array([0.0])
    print(f'{L:<8} {A_proj.mean():>+8.3f} ± {A_proj.std():.3f}          '
          f'{C_comp.mean():>+8.3f} ± {C_comp.std():.3f}          '
          f'{C_stab.mean():>+8.3f} ± {C_stab.std():.3f}')

print()
print('AUC: can competition direction separate flipped from stable?')
print(f'{"Layer":<8} {"AUC (C acts)":<20} {"AUC (A acts)"}')
print('-'*45)
for L in [7,8,9,10,11,18]:
    d = comp_directions[L]
    if comp_norms[L] < 1e-6 or not stable_pids: continue
    Cf = [acts_C[pid_to_idx_C[p],L,:] @ d for p in flipped_pids]
    Cs = [acts_C[pid_to_idx_C[p],L,:] @ d for p in stable_pids]
    Af = [acts_A[pid_to_idx_A[p],L,:] @ d for p in flipped_pids]
    As = [acts_A[pid_to_idx_A[p],L,:] @ d for p in stable_pids]
    auc_C = roc_auc_score([1]*len(Cf)+[0]*len(Cs), Cf+Cs)
    auc_A = roc_auc_score([1]*len(Af)+[0]*len(As), Af+As)
    print(f'{L:<8} {auc_C:<20.4f} {auc_A:.4f}')

np.save(str(OUT_DIR / 'competition_direction_all_layers.npy'), comp_directions)
out = {
    'n_flipped': len(flipped_pids),
    'n_stable':  len(stable_pids),
    'flipped_pids': flipped_pids,
    'direction_norms': comp_norms,
    'cosine_to_within_C': [
        float(np.dot(comp_directions[L], within_C_dirs[L]))
        if comp_norms[L]>1e-6 else 0.0 for L in range(N_LAYERS)
    ],
}
with open(OUT_DIR / 'condition_comparison_results.json','w') as f:
    json.dump(out, f, indent=2)
print('\nSaved: competition_direction_all_layers.npy')
print('Saved: condition_comparison_results.json')
