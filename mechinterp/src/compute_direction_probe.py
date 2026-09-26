"""
compute_direction_probe.py

Computes the refusal-associated direction from Condition C activations
and evaluates it via linear probing.

Design (locked):
1. Use Condition C only for direction computation (within-C refused vs complied)
   -- avoids A-vs-C condition confound
2. Stratified 70/30 train/test split by prompt_id
   -- prevents train/test leakage
3. Layer selection via 5-fold CV within C_train only
   -- no test set touches layer selection
4. Evaluate direction and probe on C_test (held out) and A (independent)
5. Primary label: replayed_label. Sensitivity: historical_label.
6. Direction called refusal-ASSOCIATED direction until causal validation.

Outputs (mechinterp/probes/):
  direction_layer{L}.npy      -- [4096] unit-norm direction vector at best layer
  probe_results.json          -- per-layer CV scores + test scores
  layer_curve_{prefill|first_tok}.png -- probe accuracy vs layer plot
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

SEED     = 42
OUT_DIR  = Path('mechinterp/probes')
ACT_DIR  = Path('mechinterp/activations')
N_LAYERS = 32
CV_FOLDS = 5

OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(cond, act_type='prefill_acts'):
    """Load activations and labels for a condition."""
    d    = np.load(ACT_DIR / f'activations_{cond}_seed42.npz')
    acts = d[act_type].astype(np.float32)        # [N, 32, 4096]
    rl   = d['replayed_labels'].astype(bool)      # primary
    hl   = d['historical_labels'].astype(bool)    # sensitivity
    meta = json.load(open(ACT_DIR / f'metadata_{cond}_seed42.json'))
    return acts, rl, hl, meta


def train_test_split_by_prompt(acts, labels, meta, test_frac=0.30):
    """
    Stratified split by prompt_id.
    Ensures no prompt appears in both train and test.
    Stratified on label to preserve class balance.
    """
    rng        = np.random.RandomState(SEED)
    prompt_ids = np.array([m['prompt_id'] for m in meta])
    unique_ids = np.unique(prompt_ids)

    # Stratify by majority label per prompt_id
    refused_ids = []
    complied_ids = []
    for pid in unique_ids:
        mask = prompt_ids == pid
        if labels[mask].mean() >= 0.5:
            refused_ids.append(pid)
        else:
            complied_ids.append(pid)

    rng.shuffle(refused_ids)
    rng.shuffle(complied_ids)

    n_test_refused  = max(1, int(len(refused_ids)  * test_frac))
    n_test_complied = max(1, int(len(complied_ids) * test_frac))

    test_ids  = set(refused_ids[:n_test_refused] +
                    complied_ids[:n_test_complied])
    train_ids = set(unique_ids) - test_ids

    train_mask = np.array([m['prompt_id'] in train_ids for m in meta])
    test_mask  = np.array([m['prompt_id'] in test_ids  for m in meta])

    return (acts[train_mask], labels[train_mask],
            acts[test_mask],  labels[test_mask])


def compute_mean_diff_direction(acts_train, labels_train, layer):
    """
    Refusal-associated direction at a given layer.
    mean(refused) - mean(complied), normalised to unit length.
    """
    X     = acts_train[:, layer, :]       # [N_train, 4096]
    ref   = X[labels_train == True]
    comp  = X[labels_train == False]
    if len(ref) == 0 or len(comp) == 0:
        return np.zeros(X.shape[1])
    direction = ref.mean(0) - comp.mean(0)
    norm      = np.linalg.norm(direction)
    return direction / norm if norm > 0 else direction


def probe_layer_cv(acts_train, labels_train, layer):
    """
    5-fold CV logistic regression probe at a given layer.
    Returns mean ROC-AUC across folds.
    """
    X  = acts_train[:, layer, :]
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=SEED)
    clf = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
    scores = cross_val_score(clf, X, labels_train,
                             cv=cv, scoring='roc_auc')
    return scores.mean(), scores.std()


def evaluate_direction(direction, acts, labels):
    """
    Project activations onto direction, compute ROC-AUC.
    Higher projection = more refusal-associated.
    """
    X          = acts[:, :, :]    # will select layer outside
    proj       = X @ direction    # [N]
    if len(np.unique(labels)) < 2:
        return {'auc': None, 'acc': None}
    auc = roc_auc_score(labels, proj)
    acc = accuracy_score(labels, proj > proj.mean())
    return {'auc': float(auc), 'acc': float(acc)}


def run_analysis(act_type):
    print(f"\n{'='*60}")
    print(f"ACTIVATION TYPE: {act_type}")
    print(f"{'='*60}")

    # Load C and A
    acts_C, rl_C, hl_C, meta_C = load_data('C', act_type)
    acts_A, rl_A, hl_A, meta_A = load_data('A', act_type)

    print(f"\nCondition C: {len(rl_C)} prompts | "
          f"refused={rl_C.sum()} complied={(~rl_C).sum()}")
    print(f"Condition A: {len(rl_A)} prompts | "
          f"refused={rl_A.sum()} complied={(~rl_A).sum()}")

    # 70/30 split on C by prompt_id
    acts_C_tr, rl_C_tr, acts_C_te, rl_C_te = \
        train_test_split_by_prompt(acts_C, rl_C, meta_C, test_frac=0.30)
    _, hl_C_tr, _, hl_C_te = \
        train_test_split_by_prompt(acts_C, hl_C, meta_C, test_frac=0.30)

    print(f"\nC_train: {len(rl_C_tr)} | "
          f"refused={rl_C_tr.sum()} complied={(~rl_C_tr).sum()}")
    print(f"C_test:  {len(rl_C_te)} | "
          f"refused={rl_C_te.sum()} complied={(~rl_C_te).sum()}")

    # ── Step 1: Layer selection via CV on C_train ─────────────────────
    print(f"\n--- Layer selection (5-fold CV on C_train) ---")
    cv_aucs = []
    for layer in range(N_LAYERS):
        mean_auc, std_auc = probe_layer_cv(acts_C_tr, rl_C_tr, layer)
        cv_aucs.append(mean_auc)
        if layer % 8 == 0 or layer == N_LAYERS - 1:
            print(f"  Layer {layer:>2}: CV AUC = {mean_auc:.3f} ± {std_auc:.3f}")

    best_layer = int(np.argmax(cv_aucs))
    print(f"\n  Best layer by CV: {best_layer} "
          f"(AUC={cv_aucs[best_layer]:.3f})")

    # ── Step 2: Compute direction at best layer from C_train ──────────
    direction = compute_mean_diff_direction(acts_C_tr, rl_C_tr, best_layer)
    np.save(OUT_DIR / f'direction_{act_type}_layer{best_layer}.npy', direction)
    print(f"\n  Direction norm: {np.linalg.norm(direction):.4f} (should be ~1.0)")
    print(f"  Saved direction to probes/direction_{act_type}_layer{best_layer}.npy")

    # ── Step 3: Evaluate direction on C_test (held out) ──────────────
    acts_C_te_layer = acts_C_te[:, best_layer, :]
    proj_C_te       = acts_C_te_layer @ direction

    if len(np.unique(rl_C_te)) >= 2:
        auc_C_te = roc_auc_score(rl_C_te, proj_C_te)
        acc_C_te = accuracy_score(rl_C_te, proj_C_te > np.median(proj_C_te))
    else:
        auc_C_te = None
        acc_C_te = None

    print(f"\n--- Q1: Does direction predict comply/refuse? ---")
    print(f"  C_test  AUC={auc_C_te:.3f}  ACC={acc_C_te:.3f}"
          if auc_C_te else "  C_test: insufficient class diversity")

    # ── Step 4: Generalisation to Condition A ────────────────────────
    acts_A_layer = acts_A[:, best_layer, :]
    proj_A       = acts_A_layer @ direction

    if len(np.unique(rl_A)) >= 2:
        auc_A = roc_auc_score(rl_A, proj_A)
        acc_A = accuracy_score(rl_A, proj_A > np.median(proj_A))
        print(f"  A (all)  AUC={auc_A:.3f}  ACC={acc_A:.3f}")
    else:
        print(f"  A: only one class ({rl_A.sum()} refused, {(~rl_A).sum()} complied)")
        print(f"  A projection: mean_refused={proj_A[rl_A].mean():.3f}  "
              f"mean_complied={proj_A[~rl_A].mean():.3f if (~rl_A).sum()>0 else 'N/A'}")

    # ── Step 5: Q2 -- Temporal analysis ──────────────────────────────
    print(f"\n--- Q2: Association with round progression (Condition C) ---")
    round_numbers = np.array([m['phase2_index'] for m in meta_C])
    proj_C_all    = (acts_C[:, best_layer, :] @ direction)
    categories    = np.array([m['category'] for m in meta_C])
    prompt_lens   = np.array([m['prompt_length'] for m in meta_C])

    # Tercile analysis controlling for category
    t1 = np.percentile(round_numbers, 33)
    t2 = np.percentile(round_numbers, 67)
    early_mask  = round_numbers <= t1
    mid_mask    = (round_numbers > t1) & (round_numbers <= t2)
    late_mask   = round_numbers > t2

    print(f"  Early rounds (idx≤{t1:.0f}): "
          f"mean_proj={proj_C_all[early_mask].mean():.3f}  "
          f"n={early_mask.sum()}  "
          f"refusal={100*rl_C[early_mask].mean():.1f}%")
    print(f"  Mid   rounds ({t1:.0f}<idx≤{t2:.0f}): "
          f"mean_proj={proj_C_all[mid_mask].mean():.3f}  "
          f"n={mid_mask.sum()}  "
          f"refusal={100*rl_C[mid_mask].mean():.1f}%")
    print(f"  Late  rounds (idx>{t2:.0f}): "
          f"mean_proj={proj_C_all[late_mask].mean():.3f}  "
          f"n={late_mask.sum()}  "
          f"refusal={100*rl_C[late_mask].mean():.1f}%")

    # Simple correlation: projection vs round_number
    corr = np.corrcoef(round_numbers, proj_C_all)[0, 1]
    print(f"\n  Pearson r(round, projection): {corr:.3f}")

    # ── Step 6: Q3 -- Condition comparison ───────────────────────────
    print(f"\n--- Q3: Generalisation of C-derived direction to A ---")
    proj_C_refused  = proj_C_all[rl_C]
    proj_C_complied = proj_C_all[~rl_C]
    proj_A_refused  = proj_A[rl_A]

    print(f"  C refused  mean proj: {proj_C_refused.mean():.3f} "
          f"± {proj_C_refused.std():.3f} (n={len(proj_C_refused)})")
    print(f"  C complied mean proj: {proj_C_complied.mean():.3f} "
          f"± {proj_C_complied.std():.3f} (n={len(proj_C_complied)})")
    print(f"  A refused  mean proj: {proj_A_refused.mean():.3f} "
          f"± {proj_A_refused.std():.3f} (n={len(proj_A_refused)})")

    # ── Step 7: Sensitivity analysis -- historical labels ─────────────
    print(f"\n--- Sensitivity: repeat Q1 with historical labels ---")
    if len(np.unique(hl_C_te)) >= 2:
        auc_hist = roc_auc_score(hl_C_te, proj_C_te)
        print(f"  C_test historical AUC={auc_hist:.3f}  "
              f"(vs replayed AUC={auc_C_te:.3f})")
    else:
        print(f"  Historical labels: only one class in C_test")

    # ── Save results ─────────────────────────────────────────────────
    results = {
        'act_type':   act_type,
        'best_layer': best_layer,
        'cv_aucs':    [float(x) for x in cv_aucs],
        'C_test_auc': float(auc_C_te) if auc_C_te else None,
        'C_test_acc': float(acc_C_te) if acc_C_te else None,
        'temporal_corr_r': float(corr),
        'tercile_early_proj': float(proj_C_all[early_mask].mean()),
        'tercile_mid_proj':   float(proj_C_all[mid_mask].mean()),
        'tercile_late_proj':  float(proj_C_all[late_mask].mean()),
        'C_refused_mean_proj':  float(proj_C_refused.mean()),
        'C_complied_mean_proj': float(proj_C_complied.mean()),
        'A_refused_mean_proj':  float(proj_A_refused.mean()),
    }
    out = OUT_DIR / f'results_{act_type}.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out}")
    return results, cv_aucs, best_layer


if __name__ == '__main__':
    print("REFUSAL-ASSOCIATED DIRECTION ANALYSIS")
    print("Primary label: replayed | Sensitivity: historical")

    all_results = {}
    for act_type in ['prefill_acts', 'first_tok_acts']:
        results, cv_aucs, best_layer = run_analysis(act_type)
        all_results[act_type] = results

    print("\n\n=== SUMMARY ===")
    for act_type, res in all_results.items():
        print(f"\n{act_type}:")
        print(f"  Best layer:     {res['best_layer']}")
        print(f"  CV AUC:         {res['cv_aucs'][res['best_layer']]:.3f}")
        print(f"  C_test AUC:     {res['C_test_auc']}")
        print(f"  Temporal r:     {res['temporal_corr_r']:.3f}")
        print(f"  C_refused proj: {res['C_refused_mean_proj']:.3f}")
        print(f"  C_complied proj:{res['C_complied_mean_proj']:.3f}")
        print(f"  A_refused proj: {res['A_refused_mean_proj']:.3f}")
