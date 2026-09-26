"""
crossfit_competition_direction.py

Fixes the in-sample bias in the competition direction AUC.

Problem: the 163 flipped prompts were used to BOTH build the direction
AND evaluate its AUC. This is in-sample evaluation.

Fix: 5-fold cross-validation on the 163 flipped prompts.
Each fold: build direction from 80% of flipped prompts,
evaluate on held-out 20% of flipped prompts + all stable prompts.

Reports: cross-validated AUC for predicting which prompts flip.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

ACT_DIR   = Path('mechinterp/activations')
PROBE_DIR = Path('mechinterp/probes')
OUT_DIR   = Path('mechinterp/results')

print('='*65)
print('CROSS-FITTED COMPETITION DIRECTION AUC')
print('Fixes in-sample bias in condition_comparison_results.json')
print('='*65)

# Load activations
data_A = np.load(str(ACT_DIR / 'activations_prehook_A_seed42.npz'))
data_C = np.load(str(ACT_DIR / 'activations_prehook_C_seed42.npz'))
acts_A = data_A['prefill_acts'].astype(np.float32)  # [676, 32, 4096]
acts_C = data_C['prefill_acts'].astype(np.float32)
rl_A   = data_A['replayed_labels'].astype(bool)
rl_C   = data_C['replayed_labels'].astype(bool)
meta_A = json.load(open(ACT_DIR / 'metadata_prehook_A_seed42.json'))
meta_C = json.load(open(ACT_DIR / 'metadata_prehook_C_seed42.json'))

pid_to_idx_A = {m['prompt_id']: i for i, m in enumerate(meta_A)}
pid_to_idx_C = {m['prompt_id']: i for i, m in enumerate(meta_C)}
common_pids  = set(pid_to_idx_A.keys()) & set(pid_to_idx_C.keys())

flipped_pids = []
stable_pids  = []
for pid in common_pids:
    a_refused = rl_A[pid_to_idx_A[pid]]
    c_refused  = rl_C[pid_to_idx_C[pid]]
    if a_refused and not c_refused:
        flipped_pids.append(pid)
    elif a_refused and c_refused:
        stable_pids.append(pid)

flipped_pids = np.array(flipped_pids)
print(f'Flipped prompts: {len(flipped_pids)}')
print(f'Stable prompts:  {len(stable_pids)}')

TARGET_LAYERS = [7, 8, 9, 10, 11, 18]
N_FOLDS = 5

print(f'\n5-fold cross-validation on {len(flipped_pids)} flipped prompts')
print(f'Each fold: direction from 80% flipped, evaluate on 20% flipped + all stable')
print()
print(f'{"Layer":<8} {"CV AUC (mean)":<18} {"CV AUC (std)":<18} {"In-sample AUC (old)"}')
print('-'*65)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
results = {}

for L in TARGET_LAYERS:
    fold_aucs = []
    all_scores = []
    all_labels = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(flipped_pids)):
        train_pids = flipped_pids[train_idx]
        test_pids  = flipped_pids[test_idx]

        # Build direction from train flipped prompts only
        A_train = np.array([acts_A[pid_to_idx_A[p], L, :] for p in train_pids])
        C_train = np.array([acts_C[pid_to_idx_C[p], L, :] for p in train_pids])
        direction = C_train.mean(0) - A_train.mean(0)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        d = direction / norm

        # Evaluate on held-out flipped (positive class) + all stable (negative class)
        test_flip_scores = [acts_C[pid_to_idx_C[p], L, :] @ d for p in test_pids]
        stable_scores    = [acts_C[pid_to_idx_C[p], L, :] @ d for p in stable_pids]

        scores = test_flip_scores + stable_scores
        labels = [1]*len(test_flip_scores) + [0]*len(stable_scores)
        auc = roc_auc_score(labels, scores)
        fold_aucs.append(auc)
        all_scores.extend(scores)
        all_labels.extend(labels)

    cv_mean = np.mean(fold_aucs)
    cv_std  = np.std(fold_aucs)

    # Also compute in-sample AUC for comparison
    A_all = np.array([acts_A[pid_to_idx_A[p], L, :] for p in flipped_pids])
    C_all = np.array([acts_C[pid_to_idx_C[p], L, :] for p in flipped_pids])
    d_all = C_all.mean(0) - A_all.mean(0)
    norm_all = np.linalg.norm(d_all)
    d_all = d_all / norm_all if norm_all > 1e-6 else d_all

    flip_scores_all   = [acts_C[pid_to_idx_C[p], L, :] @ d_all for p in flipped_pids]
    stable_scores_all = [acts_C[pid_to_idx_C[p], L, :] @ d_all for p in stable_pids]
    insample_auc = roc_auc_score(
        [1]*len(flip_scores_all) + [0]*len(stable_scores_all),
        flip_scores_all + stable_scores_all
    )

    results[L] = {'cv_mean': cv_mean, 'cv_std': cv_std, 'insample': insample_auc,
                  'fold_aucs': fold_aucs}
    print(f'{L:<8} {cv_mean:<18.4f} {cv_std:<18.4f} {insample_auc:.4f}')

print()
print('Key comparison:')
print(f'  Layer 18 in-sample AUC:      {results[18]["insample"]:.4f}  (reported before, vulnerable to circularity)')
print(f'  Layer 18 cross-fitted AUC:   {results[18]["cv_mean"]:.4f} ± {results[18]["cv_std"]:.4f}  (defensible out-of-sample estimate)')

out = {'target_layers': TARGET_LAYERS, 'n_folds': N_FOLDS,
       'n_flipped': len(flipped_pids), 'n_stable': len(stable_pids),
       'results': {str(L): results[L] for L in TARGET_LAYERS}}
with open(OUT_DIR / 'crossfit_competition_auc.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f'\nSaved: mechinterp/results/crossfit_competition_auc.json')
