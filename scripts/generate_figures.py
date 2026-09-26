"""
generate_figures.py
===================
Self-contained figure generation for CSS v3 behavioural + mechanistic results.

Run from the project root:
    cd ~/compete_safety_v2
    python3 scripts/generate_figures.py

Outputs saved to: figures/ directory (created if absent)

Data loading strategy:
  1. Load directly from result files on disk (confirmed paths from codebase)
  2. Fall back to hardcoded values from conversation history
  All hardcoded values marked with # HARDCODED: <source>
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from collections import defaultdict

OUT = Path('figures')
OUT.mkdir(exist_ok=True)
print(f'Output directory: {OUT.resolve()}')

plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.size':         13,
    'axes.titlesize':    14,
    'axes.labelsize':    13,
    'xtick.labelsize':   12,
    'ytick.labelsize':   12,
    'legend.fontsize':   12,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.facecolor': 'white',
})

BLUE   = '#2a78d6'
ORANGE = '#eda100'
RED    = '#eb6834'
GREEN  = '#1baf7a'
PURPLE = '#8b1a8b'
GRAY   = '#777777'
SEEDS  = [42, 123, 777]

# ── helpers ────────────────────────────────────────────────────────────

def load_phase2(path):
    recs = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get('phase') == 'phase2':
                recs.append(r)
    return recs

def refusal_rate_single(records):
    if not records: return None
    return sum(1 for r in records if r.get('is_safe_honest', True)) / len(records)

def refusal_rate_zeta(records):
    if not records: return None
    return sum(1 for r in records
               if r.get('zeta', {}).get('is_safe_honest', True)) / len(records)

def try_load(path):
    p = Path(path)
    if p.exists():
        return load_phase2(p)
    print(f'  Not found: {path}')
    return None

# ══════════════════════════════════════════════════════════════════════
# FIG 1 — SINGLE-AGENT A/B/C
# ══════════════════════════════════════════════════════════════════════
print('\n─── Fig 1: Single-agent A/B/C ───')

BASE = Path('results_task6_676prompt')
COND_MAP = {
    'A': 'control',
    'B': 'competition_no_memory',
    'C': 'competition_with_memory',
}

# HARDCODED: from rounds.jsonl McNemar analysis in conversation
SINGLE_HC = {
    'A': [98.5, 99.1, 99.1],
    'B': [89.4, 85.6, 85.6],
    'C': [78.0, 76.5, 76.5],
}

single_rates = {}
for cond, name in COND_MAP.items():
    rates = []
    for seed in SEEDS:
        recs = try_load(BASE / f'{name}_seed{seed}' / 'rounds.jsonl')
        if recs is not None:
            rates.append(refusal_rate_single(recs) * 100)
    single_rates[cond] = rates if rates else SINGLE_HC[cond]
    src = 'disk' if rates else 'hardcoded'
    print(f'  {cond}: {single_rates[cond]} [{src}]')

fig, ax = plt.subplots(figsize=(8, 6))
labels = ['A: Control', 'B: Competition\nno memory', 'C: Competition\nwith memory']
colors = [BLUE, ORANGE, RED]
x = np.arange(3)
w = 0.55

for i, (cond, color) in enumerate(zip(['A','B','C'], colors)):
    vals = single_rates[cond]
    mean, std = np.mean(vals), np.std(vals)
    ax.bar(i, mean, w, color=color, alpha=0.82,
           yerr=std, capsize=7, error_kw={'linewidth':2,'color':'#333'})
    for v in vals:
        ax.plot(i, v, 'o', color='black', markersize=7, zorder=6)
    ax.text(i, mean+std+1.5, f'{mean:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel('Mean refusal rate (%)', fontsize=13)
ax.set_ylim(60, 108)
ax.set_title('Refusal Rate Across Experimental Conditions', fontsize=14,
             fontweight='bold', pad=10)
ax.text(0.5, 1.01,
        'Llama 3.1 8B Instruct · 676 prompts · seeds 42/123/777',
        transform=ax.transAxes, ha='center', fontsize=11, color=GRAY)
ax.text(0.98, 0.04,
        'Bars = mean · Dots = individual seeds · Error bars = ±1 SD',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
        color=GRAY, bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', edgecolor='#ddd'))
# Significance annotation
for x1, x2, y, label in [(0,1,104,'OR=17, p<10⁻⁸'),
                           (1,2,101,'OR=14, p<10⁻⁸'),
                           (0,2,107,'OR=30, p<10⁻⁸')]:
    ax.annotate('', xy=(x2,y), xytext=(x1,y),
                arrowprops=dict(arrowstyle='-', color='#333', lw=1.2))
    ax.text((x1+x2)/2, y+0.2, label, ha='center', fontsize=8.5, color='#333')

plt.tight_layout()
fig.savefig(OUT / 'fig1_single_agent_ABC.png')
plt.close()
print('  Saved: fig1_single_agent_ABC.png')


# ══════════════════════════════════════════════════════════════════════
# FIG 2 — ROLLING REFUSAL RATE OVER TOURNAMENT (A/B/C)
# ══════════════════════════════════════════════════════════════════════
print('\n─── Fig 2: Rolling refusal rate over tournament ───')

def rolling_refusal(records, window=50):
    labels = [1 if r.get('is_safe_honest', True) else 0 for r in records]
    if len(labels) < window: return [], []
    rates = [np.mean(labels[max(0,i-window):i+1]) for i in range(len(labels))]
    return list(range(len(rates))), rates

cond_colors = {'A': BLUE, 'B': ORANGE, 'C': RED}
cond_fullnames = {'A':'A — Control', 'B':'B — No memory', 'C':'C — With memory'}

fig, ax = plt.subplots(figsize=(11, 5.5))
for cond, name in COND_MAP.items():
    all_y = []
    for seed in SEEDS:
        recs = try_load(BASE / f'{name}_seed{seed}' / 'rounds.jsonl')
        if recs:
            x_r, y_r = rolling_refusal(recs, window=50)
            ax.plot(x_r, [yi*100 for yi in y_r],
                    color=cond_colors[cond], alpha=0.25, linewidth=0.9)
            all_y.append(y_r)
    if all_y:
        ml = min(len(y) for y in all_y)
        my = np.mean([y[:ml] for y in all_y], axis=0)
        ax.plot(range(ml), [yi*100 for yi in my],
                color=cond_colors[cond], linewidth=2.5,
                label=cond_fullnames[cond])
    else:
        # HARDCODED: approximate mean rolling rates from conversation
        approx = {'A': [(0,99),(676,99)],
                  'B': [(0,92),(50,87),(300,87),(500,88),(676,87)],
                  'C': [(0,90),(50,79),(200,76),(400,74),(676,77)]}
        xs = [p[0] for p in approx[cond]]
        ys = [p[1] for p in approx[cond]]
        ax.plot(xs, ys, color=cond_colors[cond], linewidth=2.5,
                linestyle='--', label=cond_fullnames[cond]+' (approx)')

ax.set_xlabel('Phase 2 round', fontsize=13)
ax.set_ylabel('Rolling refusal rate (%) — window 50', fontsize=13)
ax.set_ylim(55, 105)
ax.set_title('Rolling Refusal Rate Over Tournament',
             fontsize=14, fontweight='bold', pad=10)
ax.text(0.5, 1.01,
        'Thin lines = individual seeds · Thick lines = mean across seeds',
        transform=ax.transAxes, ha='center', fontsize=11, color=GRAY)
ax.legend(fontsize=12, loc='lower left')
plt.tight_layout()
fig.savefig(OUT / 'fig2_rolling_refusal_ABC.png')
plt.close()
print('  Saved: fig2_rolling_refusal_ABC.png')


# ══════════════════════════════════════════════════════════════════════
# FIG 3 — CATEGORY-LEVEL COMPLIANCE LIFT
# ══════════════════════════════════════════════════════════════════════
print('\n─── Fig 3: Category compliance lift ───')

cat_A = defaultdict(list)
cat_C = defaultdict(list)
cat_loaded = False

for seed in SEEDS:
    for cond, store in [('control', cat_A),
                        ('competition_with_memory', cat_C)]:
        recs = try_load(BASE / f'{cond}_seed{seed}' / 'rounds.jsonl')
        if recs:
            for r in recs:
                cat = r.get('category','unknown')
                if cat not in ('unknown', None, ''):
                    store[cat].append(
                        1 if r.get('is_safe_honest', True) else 0)
            cat_loaded = True

if cat_loaded and cat_A and cat_C:
    cat_increases = {}
    for cat in set(cat_A) & set(cat_C):
        ra = np.mean(cat_A[cat]) * 100
        rc = np.mean(cat_C[cat]) * 100
        cat_increases[cat] = rc - ra  # negative = MORE compliance in C
    cat_increases = {k: -v for k, v in cat_increases.items()}
else:
    # HARDCODED: Fisher's exact test A→C pooled across seeds
    cat_increases = {
        'Cybercrime / intrusion': 42.6,
        'Harmful content':        33.3,
        'Chemical / biological':  30.6,
        'Illegal activities':     18.9,
        'Misinformation':         11.1,
        'Harassment / bullying':   0.0,
    }
    print('  Using hardcoded category values')

cats_sorted = sorted(cat_increases.items(), key=lambda x: x[1], reverse=True)
cat_names = [c[0] for c in cats_sorted]
cat_vals  = [c[1] for c in cats_sorted]

fig, ax = plt.subplots(figsize=(10, 5.5))
y = np.arange(len(cat_names))
bars = ax.barh(y, cat_vals, height=0.6,
               color=[RED if v > 0 else GRAY for v in cat_vals], alpha=0.85)
for bar, val in zip(bars, cat_vals):
    label = f'+{val:.1f}pp' if val > 0 else '0pp'
    ax.text(val+0.5, bar.get_y()+bar.get_height()/2,
            label, va='center', fontsize=12, fontweight='bold')
ax.set_yticks(y); ax.set_yticklabels(cat_names, fontsize=12)
ax.set_xlabel('Compliance increase A→C (percentage points)', fontsize=13)
ax.set_xlim(0, 55)
ax.set_title('Compliance Increase by Harm Category (Condition A to C)',
             fontsize=14, fontweight='bold', pad=10)
ax.text(0.5, 1.01, "Pooled across seeds · Fisher's exact test",
        transform=ax.transAxes, ha='center', fontsize=11, color=GRAY)
ax.invert_yaxis()
ax.axvline(0, color='#333', linewidth=0.8)
plt.tight_layout()
fig.savefig(OUT / 'fig3_category_compliance_lift.png')
plt.close()
print('  Saved: fig3_category_compliance_lift.png')


# ══════════════════════════════════════════════════════════════════════
# FIG 4 — ESCALATION CONDITIONS (E0/E1/E2/E3)
# ══════════════════════════════════════════════════════════════════════
print('\n─── Fig 4: Escalation conditions ───')

ESC_MAP = {
    'E0 — Flat (C)':       ('results_task6_676prompt',
                             'competition_with_memory'),
    'E1 — Fixed ratio':    ('results_escalation_E1',
                             'competition_with_memory'),
    'E2 — Asymmetric':     ('results_escalation_E2',
                             'competition_with_memory'),
    'E3 — Loss dominant':  ('results_escalation_E3',
                             'competition_with_memory'),
}
# HARDCODED: from escalation null result reported in conversation
ESC_HC = {
    'E0 — Flat (C)':      [77.6, 76.1, 78.2],
    'E1 — Fixed ratio':   [78.5, 77.9, 78.2],
    'E2 — Asymmetric':    [79.1, 77.4, 78.2],
    'E3 — Loss dominant': [77.4, 77.1, 78.3],
}
ESC_COLORS = [BLUE, ORANGE, RED, PURPLE]

esc_rates = {}
for label, (dname, prefix) in ESC_MAP.items():
    rates = []
    for seed in SEEDS:
        recs = try_load(Path(dname) / f'{prefix}_seed{seed}' / 'rounds.jsonl')
        if recs:
            rates.append(refusal_rate_single(recs) * 100)
    esc_rates[label] = rates if rates else ESC_HC[label]
    src = 'disk' if rates else 'hardcoded'
    print(f'  {label}: {[f"{v:.1f}" for v in esc_rates[label]]} [{src}]')

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: bar chart
ax = axes[0]
esc_labels = list(ESC_MAP.keys())
x = np.arange(len(esc_labels))
w = 0.55
for i, (label, color) in enumerate(zip(esc_labels, ESC_COLORS)):
    vals = esc_rates[label]
    mean, std = np.mean(vals), np.std(vals)
    ax.bar(i, mean, w, color=color, alpha=0.82,
           yerr=std, capsize=6, error_kw={'linewidth':1.8,'color':'#333'})
    for v in vals:
        ax.plot(i, v, 'o', color='black', markersize=6, zorder=6)
    ax.text(i, mean+std+0.8, f'{mean:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['E0\nFlat','E1\nFixed','E2\nAsym.','E3\nLoss'],
                   fontsize=11)
ax.set_ylabel('Mean refusal rate (%)', fontsize=12)
ax.set_ylim(65, 90)
ax.set_title('Mean Refusal by Penalty Schedule', fontsize=13, fontweight='bold')

# Right: rolling lines
ax2 = axes[1]
for label, (dname, prefix), color in zip(esc_labels, ESC_MAP.values(), ESC_COLORS):
    all_y = []
    for seed in SEEDS:
        recs = try_load(Path(dname) / f'{prefix}_seed{seed}' / 'rounds.jsonl')
        if recs:
            xr, yr = rolling_refusal(recs, window=50)
            ax2.plot(xr, [yi*100 for yi in yr],
                     color=color, alpha=0.2, linewidth=0.9)
            all_y.append(yr)
    if all_y:
        ml = min(len(y) for y in all_y)
        my = np.mean([y[:ml] for y in all_y], axis=0)
        ax2.plot(range(ml), [yi*100 for yi in my],
                 color=color, linewidth=2.2, label=label)
ax2.set_xlabel('Phase 2 round', fontsize=12)
ax2.set_ylabel('Rolling refusal rate (%) — window 50', fontsize=12)
ax2.set_ylim(55, 100)
ax2.set_title('Rolling Refusal Over Tournament', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10, loc='lower left')

fig.suptitle('Escalation: Effect of Penalty Schedule on Refusal Rate\n'
             'NULL RESULT — all schedules produce 75–79% refusal',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(OUT / 'fig4_escalation.png')
plt.close()
print('  Saved: fig4_escalation.png')


# ══════════════════════════════════════════════════════════════════════
# FIG 5 — TWO-AGENT PEER VISIBILITY (I0/I2_FULL/I2_COMP)
# ══════════════════════════════════════════════════════════════════════
print('\n─── Fig 5: Two-agent peer visibility ───')

TWO_MAP = {
    'I0':      ('results_two_agent_I0',
                'two_agent_I0_competition_with_memory'),
    'I2_FULL': ('results_two_agent_I2_FULL',
                'two_agent_I2_I3_FULL_competition_with_memory'),
    'I2_COMP': ('results_two_agent_I2_COMPLIANCE_ONLY',
                'two_agent_I2_I3_COMPLIANCE_ONLY_competition_with_memory'),
}
# HARDCODED: from corrected two-agent analysis in conversation
TWO_HC = {
    'I0':      [77.6, 76.1, 78.2],
    'I2_FULL': [84.7, 83.7, 83.1],
    'I2_COMP': [69.2, 68.8, 73.9],
}
# HARDCODED: Sigma refusal rates confirmed in conversation
SIGMA_HC = {
    'I0':      [90.2, 88.6, 89.0],
    'I2_FULL': [91.4, 91.0, 90.8],
    'I2_COMP': [91.4, 92.0, 89.4],
}
TWO_COLORS = [BLUE, GREEN, RED]

two_rates   = {}
two_rolling = {}

for cond, (dname, prefix) in TWO_MAP.items():
    rates = []; all_y = []
    for seed in SEEDS:
        recs = try_load(Path(dname) / f'{prefix}_seed{seed}' / 'rounds.jsonl')
        if recs:
            rates.append(refusal_rate_zeta(recs) * 100)
            xr, yr = rolling_refusal(recs, window=30)
            # rolling on zeta field
            labels_z = [1 if r.get('zeta',{}).get('is_safe_honest',True)
                        else 0 for r in recs]
            yr_z = [np.mean(labels_z[max(0,i-30):i+1])
                    for i in range(len(labels_z))]
            all_y.append(yr_z)
    two_rates[cond]   = rates if rates else TWO_HC[cond]
    two_rolling[cond] = all_y
    src = 'disk' if rates else 'hardcoded'
    print(f'  {cond}: {[f"{v:.1f}" for v in two_rates[cond]]} [{src}]')

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: bar chart
ax = axes[0]
two_xlabels = ['I0\n(No visibility)', 'I2_FULL\n(Full visibility)',
               'I2_COMP\n(Comply only)']
x = np.arange(3)
for i, (cond, color) in enumerate(
        zip(['I0','I2_FULL','I2_COMP'], TWO_COLORS)):
    vals = two_rates[cond]
    mean, std = np.mean(vals), np.std(vals)
    ax.bar(i, mean, 0.55, color=color, alpha=0.82,
           yerr=std, capsize=6, error_kw={'linewidth':1.8,'color':'#333'})
    for v in vals:
        ax.plot(i, v, 'o', color='black', markersize=6, zorder=6)
    ax.text(i, mean+std+0.8, f'{mean:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

sigma_all = [v for vals in SIGMA_HC.values() for v in vals]
sigma_mean = np.mean(sigma_all)
ax.axhline(sigma_mean, color=GRAY, linestyle='--', linewidth=1.8, alpha=0.8)
ax.text(2.4, sigma_mean+0.6, f'Σ Sigma ≈{sigma_mean:.0f}%',
        fontsize=10, color=GRAY, ha='right')

ax.set_xticks(x); ax.set_xticklabels(two_xlabels, fontsize=11)
ax.set_ylabel('Zeta refusal rate (%)', fontsize=12)
ax.set_ylim(55, 100)
ax.set_title('Refusal by Peer Visibility', fontsize=13, fontweight='bold')

# Right: rolling lines
ax2 = axes[1]
for cond, color in zip(['I0','I2_FULL','I2_COMP'], TWO_COLORS):
    all_y = two_rolling[cond]
    if all_y:
        for y in all_y:
            ax2.plot(range(len(y)), [yi*100 for yi in y],
                     color=color, alpha=0.2, linewidth=0.9)
        ml = min(len(y) for y in all_y)
        my = np.mean([y[:ml] for y in all_y], axis=0)
        ax2.plot(range(ml), [yi*100 for yi in my],
                 color=color, linewidth=2.2, label=cond)
ax2.set_xlabel('Phase 2 round', fontsize=12)
ax2.set_ylabel('Zeta rolling refusal (%) — window 30', fontsize=12)
ax2.set_ylim(50, 100)
ax2.set_title('Rolling Refusal Over Tournament', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10, loc='lower left')

fig.suptitle('Effect of Competitor Visibility on Zeta Refusal Rate\n'
             'Zeta: Llama 3.1 8B Instruct · Sigma: Qwen3 8B · 510 prompts · 3 seeds',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(OUT / 'fig5_two_agent_visibility.png')
plt.close()
print('  Saved: fig5_two_agent_visibility.png')


# ══════════════════════════════════════════════════════════════════════
# FIG 6 — MECHANISTIC: CV AUC PROFILE ACROSS LAYERS
# ══════════════════════════════════════════════════════════════════════
print('\n─── Fig 6: CV AUC profile ───')

# HARDCODED: from compute_direction_probe.py output in conversation
AUC_HC = {
    0:0.500,1:0.601,2:0.664,3:0.682,4:0.679,5:0.682,6:0.666,
    7:0.699,8:0.769,9:0.804,10:0.875,11:0.925,12:0.936,13:0.955,
    14:0.960,15:0.960,16:0.971,17:0.974,18:0.975,19:0.975,
    20:0.974,21:0.974,22:0.973,23:0.972,24:0.973,25:0.972,
    26:0.972,27:0.973,28:0.973,29:0.973,30:0.973,31:0.973,
}
# HARDCODED: from layer_specific_ablation_results.json in conversation
FLIPS_HC = {
    0:0,1:0,2:0,3:10,4:5,5:2,6:3,
    7:26,8:31,9:60,10:38,11:57,
    12:9,13:13,14:30,
    15:0,16:2,17:8,18:2,19:1,20:0,21:2,
    22:0,23:0,24:0,25:0,26:0,27:1,28:1,29:0,30:0,31:0,
}

# Try loading from disk
auc_data = AUC_HC.copy()
flip_data = FLIPS_HC.copy()
auc_src = 'hardcoded'
flip_src = 'hardcoded'

auc_file  = Path('mechinterp/results/layer_auc_profile.json')
flip_file = Path('mechinterp/results/layer_specific_ablation_results.json')

if auc_file.exists():
    raw = json.load(open(auc_file))
    auc_data = {int(k): v for k, v in raw.items()}
    auc_src = 'disk'
    print(f'  Loaded AUC from disk')

if flip_file.exists():
    raw = json.load(open(flip_file))
    if 'layers' in raw:
        flip_data = {}
        for ld in raw['layers']:
            if not ld.get('skipped'):
                flip_data[ld['layer']] = ld.get('R_to_C', 0)
        flip_src = 'disk'
        print(f'  Loaded ablation flips from disk')

layers_all = list(range(32))
aucs_all   = [auc_data.get(L, 0.5) for L in layers_all]
flips_all  = [flip_data.get(L, 0) for L in layers_all]

fig, ax1 = plt.subplots(figsize=(12, 5.5))
bars = ax1.bar(layers_all, flips_all, color=BLUE, alpha=0.72, width=0.75,
               label='R→C flips (ablation, left axis)')
ax1.set_xlabel('Transformer layer', fontsize=13)
ax1.set_ylabel('R→C flips out of 153 (bars)', fontsize=13, color=BLUE)
ax1.tick_params(axis='y', labelcolor=BLUE)
ax1.set_ylim(0, 75)
ax1.set_xlim(-0.8, 31.8)

for L, f in [(9, flips_all[9]), (11, flips_all[11])]:
    if f > 0:
        ax1.text(L, f+1.5, str(f), ha='center', va='bottom',
                 fontsize=10, fontweight='bold', color=BLUE)

ax2 = ax1.twinx()
ax2.plot(layers_all, aucs_all, 'o-', color=RED, linewidth=2.2,
         markersize=4, label='CV AUC (right axis)', alpha=0.88)
ax2.set_ylabel('CV AUC — representational strength', fontsize=13, color=RED)
ax2.tick_params(axis='y', labelcolor=RED)
ax2.set_ylim(0.45, 1.02)
ax2.axhline(0.975, color=RED, linestyle=':', alpha=0.35, linewidth=1)

ax1.axvspan(7, 14, alpha=0.07, color=BLUE)
ax1.axvspan(16, 31, alpha=0.05, color=RED)
ax1.text(10.5, 68, 'Peak causal\nzone (L7–14)',
         ha='center', fontsize=9.5, color=BLUE, alpha=0.8)
ax1.text(23, 68, 'High repr., low\ncausal effect',
         ha='center', fontsize=9.5, color=RED, alpha=0.8)

lines1, lbl1 = ax1.get_legend_handles_labels()
lines2, lbl2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, lbl1+lbl2, loc='upper left', fontsize=10)

ax1.set_title('Layer-Specific Ablation Flips vs Representational Strength (CV AUC)',
              fontsize=14, fontweight='bold', pad=10)
ax1.text(0.5, 1.01,
         'Llama 3.1 8B Instruct · 153 C_test refused prompts · seed 42',
         transform=ax1.transAxes, ha='center', fontsize=11, color=GRAY)

note = f'AUC: {auc_src} · Flips: {flip_src}'
if 'hardcoded' in note:
    ax1.text(0.02, 0.02, f'* {note} (files not found on disk)',
             transform=ax1.transAxes, fontsize=9, color=GRAY, style='italic')

plt.tight_layout()
fig.savefig(OUT / 'fig6_ablation_auc_dissociation.png')
plt.close()
print('  Saved: fig6_ablation_auc_dissociation.png')


# ══════════════════════════════════════════════════════════════════════
# FIG 7 — MECHANISTIC: PAIRED A→C SHIFT (FLIP vs STABLE)
# ══════════════════════════════════════════════════════════════════════
print('\n─── Fig 7: Paired A→C shift ───')

SHIFT_FILE = Path('mechinterp/results/paired_shift_analysis.json')
shift_loaded = False

if SHIFT_FILE.exists():
    raw = json.load(open(SHIFT_FILE))
    layers_sh, flip_sh, stab_sh, fstd_sh, sstd_sh = [], [], [], [], []
    for L_str in sorted(raw, key=lambda x: int(x)):
        d = raw[L_str]
        layers_sh.append(int(L_str))
        flip_sh.append(d['flip_mean'])
        stab_sh.append(d['stab_mean'])
        fstd_sh.append(d['flip_std'])
        sstd_sh.append(d['stab_std'])
    shift_loaded = True
    print(f'  Loaded from disk: {len(layers_sh)} layers')
else:
    # HARDCODED: from paired_shift_analysis.py output in conversation
    layers_sh = [7,    8,    9,    10,   11,   14,    18,     24,      31]
    flip_sh   = [-0.356,-0.404,-0.227,-0.225,-0.551,-2.171,-6.022,-13.066,-25.937]
    stab_sh   = [-0.340,-0.408,-0.270,-0.237,-0.541,-0.845,-1.978, -2.937, -3.970]
    fstd_sh   = [0.100, 0.139, 0.173, 0.214, 0.251, 0.599,  1.226,  2.510,  4.738]
    sstd_sh   = [0.129, 0.171, 0.205, 0.232, 0.279, 0.709,  1.422,  3.285,  6.912]
    print('  Using hardcoded values (9 layers)')

n_flip = 163; n_stab = 512  # HARDCODED: confirmed throughout conversation
f_sem = np.array(fstd_sh) / np.sqrt(n_flip)
s_sem = np.array(sstd_sh) / np.sqrt(n_stab)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(layers_sh, flip_sh, 'o-', color=RED, linewidth=2.2, markersize=8,
        label=f'A→C flip prompts (n={n_flip})')
ax.fill_between(layers_sh,
                np.array(flip_sh)-f_sem, np.array(flip_sh)+f_sem,
                color=RED, alpha=0.15)
ax.plot(layers_sh, stab_sh, 's--', color=BLUE, linewidth=2.2, markersize=7,
        label=f'Stable refusal prompts (n={n_stab})')
ax.fill_between(layers_sh,
                np.array(stab_sh)-s_sem, np.array(stab_sh)+s_sem,
                color=BLUE, alpha=0.12)
ax.axhline(0, color='#333', linewidth=0.9, alpha=0.5)

if 18 in layers_sh:
    i18 = layers_sh.index(18)
    ax.annotate(f'L18: d=−3.046\np=5.5×10⁻⁷³',
                xy=(18, flip_sh[i18]),
                xytext=(21, -9),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.3),
                fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='#fff3f3', edgecolor='#ccc'))

if 9 in layers_sh:
    ax.annotate('L7–11: groups\nindistinguishable\n(all p>0.17)',
                xy=(9, stab_sh[layers_sh.index(9)]),
                xytext=(3, -5),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.3),
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='#f3f6ff', edgecolor='#ccc'))

ax.set_xlabel('Transformer layer', fontsize=13)
ax.set_ylabel('Mean shift projected onto refusal direction', fontsize=13)
ax.set_title('Activation Projection onto Refusal Direction by Layer',
             fontsize=14, fontweight='bold', pad=10)
ax.text(0.5, 1.01,
        f'163 A→C flip prompts vs {n_stab} stable refusal prompts — seed 42',
        transform=ax.transAxes, ha='center', fontsize=11, color=GRAY)
ax.legend(fontsize=12, loc='lower left')
ax.set_xticks(layers_sh)

if not shift_loaded:
    ax.text(0.02, 0.02,
            f'* 9-layer subset from conversation · full data: {SHIFT_FILE}',
            transform=ax.transAxes, fontsize=9, color=GRAY, style='italic')

plt.tight_layout()
fig.savefig(OUT / 'fig7_paired_shift_flip_vs_stable.png')
plt.close()
print('  Saved: fig7_paired_shift_flip_vs_stable.png')


# ══════════════════════════════════════════════════════════════════════
# FIG 8 — MECHANISTIC: REFUSAL RESTORATION
# ══════════════════════════════════════════════════════════════════════
print('\n─── Fig 8: Refusal restoration ───')

RESTORE_FILE = Path('mechinterp/results/refusal_restoration_results.json')
restore_loaded = False

if RESTORE_FILE.exists():
    raw = json.load(open(RESTORE_FILE))
    exp3 = raw.get('exp3_restoration', {})
    restore_loaded = True
    print('  Loaded restoration results from disk')
else:
    # HARDCODED: from refusal_restoration_results.json output in conversation
    exp3 = {
        '9': {
            '0.5': {'baseline_refusal':0.0,'learned_refusal':0.681,'random_refusal':0.061},
            '1.0': {'baseline_refusal':0.0,'learned_refusal':0.914,'random_refusal':0.067},
            '2.0': {'baseline_refusal':0.0,'learned_refusal':1.000,'random_refusal':0.037},
            '5.0': {'baseline_refusal':0.0,'learned_refusal':0.515,'random_refusal':0.043},
        },
        '11': {
            '0.5': {'baseline_refusal':0.0,'learned_refusal':0.669,'random_refusal':0.037},
            '1.0': {'baseline_refusal':0.0,'learned_refusal':0.951,'random_refusal':0.037},
            '2.0': {'baseline_refusal':0.0,'learned_refusal':1.000,'random_refusal':0.031},
            '5.0': {'baseline_refusal':0.0,'learned_refusal':0.969,'random_refusal':0.012},
        },
        '18': {
            '0.5': {'baseline_refusal':0.0,'learned_refusal':0.129,'random_refusal':0.031},
            '1.0': {'baseline_refusal':0.0,'learned_refusal':0.264,'random_refusal':0.043},
            '2.0': {'baseline_refusal':0.0,'learned_refusal':0.546,'random_refusal':0.067},
            '5.0': {'baseline_refusal':0.0,'learned_refusal':1.000,'random_refusal':0.061},
        },
    }
    print('  Using hardcoded restoration values')

ALPHAS = [0.5, 1.0, 2.0, 5.0]
layer_colors = {'9': RED, '11': ORANGE, '18': GREEN}
layer_styles  = {'9': '-', '11': '--', '18': '-.'}

fig, ax = plt.subplots(figsize=(8, 5.5))
for L, color, style in [('9',RED,'-'),('11',ORANGE,'--'),('18',GREEN,'-.')]:
    if L not in exp3: continue
    learned = [exp3[L][str(a)]['learned_refusal']*100 for a in ALPHAS]
    random  = [exp3[L][str(a)]['random_refusal']*100  for a in ALPHAS]
    ax.plot(ALPHAS, learned, style, color=color, linewidth=2.2, marker='o',
            markersize=8, label=f'Layer {L} (learned direction)')
    ax.plot(ALPHAS, random, style, color=color, linewidth=1.0, marker='x',
            markersize=8, alpha=0.45, markeredgewidth=2)

ax.axhline(0, color='#333', linewidth=0.8, linestyle=':', alpha=0.6)
ax.text(5.05, 1.5, 'Baseline\n(0%)', fontsize=9, color='#555')
ax.text(0.98, 0.06,
        'Solid = learned direction\n× = random orthogonal control',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=10, color=GRAY)
ax.set_xlabel('Injection strength (α)', fontsize=13)
ax.set_ylabel('Refusal rate after direction injection (%)', fontsize=13)
ax.set_ylim(-5, 108)
ax.set_xlim(0.3, 5.5)
ax.set_title('Refusal Restoration Under Competition', fontsize=14,
             fontweight='bold', pad=10)
ax.text(0.5, 1.01,
        '163 C_test complied prompts — full competitive context active — seed 42',
        transform=ax.transAxes, ha='center', fontsize=11, color=GRAY)
ax.legend(fontsize=11, loc='upper right')

if not restore_loaded:
    ax.text(0.02, 0.02,
            f'* Values from conversation · full data: {RESTORE_FILE}',
            transform=ax.transAxes, fontsize=9, color=GRAY, style='italic')

plt.tight_layout()
fig.savefig(OUT / 'fig8_refusal_restoration.png')
plt.close()
print('  Saved: fig8_refusal_restoration.png')


# ══════════════════════════════════════════════════════════════════════
# FIG 9 — ACTIVATION PATCHING HEATMAP (if data exists)
# ══════════════════════════════════════════════════════════════════════
print('\n─── Fig 9: Activation patching heatmap ───')

PATCH_FILE = Path('mechinterp/results/activation_patching_results.json')

if PATCH_FILE.exists():
    patch_data = json.load(open(PATCH_FILE))
    directions = list(patch_data.keys())
    all_layers = sorted(set(
        int(L) for d in patch_data.values() for L in d.keys()))
    matrix = np.full((len(directions), len(all_layers)), np.nan)
    for di, direction in enumerate(directions):
        for li, L in enumerate(all_layers):
            val = patch_data[direction].get(str(L),
                  patch_data[direction].get(L, np.nan))
            matrix[di, li] = val

    fig, ax = plt.subplots(figsize=(max(8, len(all_layers)*0.8), 4))
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-100, vmax=100, aspect='auto')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('% prompts behaviour changed', fontsize=12)
    ax.set_xticks(range(len(all_layers)))
    ax.set_xticklabels([str(L) for L in all_layers], fontsize=11)
    ax.set_yticks(range(len(directions)))
    ax.set_yticklabels(directions, fontsize=12)
    ax.set_xlabel('Transformer layer', fontsize=13)
    ax.set_title('Activation Patching Results by Layer and Direction',
                 fontsize=14, fontweight='bold', pad=10)
    ax.text(0.5, 1.01, '163 matched A→C flip prompts',
            transform=ax.transAxes, ha='center', fontsize=11, color=GRAY)
    for di in range(len(directions)):
        for li in range(len(all_layers)):
            v = matrix[di, li]
            if not np.isnan(v):
                ax.text(li, di, f'{v:.1f}', ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        color='white' if abs(v) > 50 else '#222')
    plt.tight_layout()
    fig.savefig(OUT / 'fig9_activation_patching.png')
    plt.close()
    print('  Saved: fig9_activation_patching.png')

else:
    print()
    print('  ╔══════════════════════════════════════════════════════════╗')
    print('  ║  FIG 9 CANNOT BE GENERATED — EXPERIMENT NOT YET RUN     ║')
    print('  ╠══════════════════════════════════════════════════════════╣')
    print('  ║  Activation patching = gold-standard causal test        ║')
    print('  ║  (patch A representations into C forward passes)        ║')
    print('  ║  No results exist yet.                                   ║')
    print('  ║                                                          ║')
    print('  ║  Missing values:                                         ║')
    print('  ║    For layers 9, 11, 14, 18, 24:                        ║')
    print('  ║      A→C: % prompts that flip to compliance after patch ║')
    print('  ║      C→A: % prompts that flip to refusal after patch    ║')
    print('  ║                                                          ║')
    print(f'  ║  Save results to: {str(PATCH_FILE):<38}║')
    print('  ║  Structure: {"A_to_C":{"9":45.2,...},"C_to_A":{...}}   ║')
    print('  ║  Then re-run this script.                                ║')
    print('  ╚══════════════════════════════════════════════════════════╝')
    print()


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print('\n═══════════════════════════════════════════════════')
print('SUMMARY')
print('═══════════════════════════════════════════════════')
expected = [
    'fig1_single_agent_ABC.png',
    'fig2_rolling_refusal_ABC.png',
    'fig3_category_compliance_lift.png',
    'fig4_escalation.png',
    'fig5_two_agent_visibility.png',
    'fig6_ablation_auc_dissociation.png',
    'fig7_paired_shift_flip_vs_stable.png',
    'fig8_refusal_restoration.png',
    'fig9_activation_patching.png',
]
saved = 0
for f in expected:
    p = OUT / f
    if p.exists():
        kb = p.stat().st_size / 1024
        print(f'  ✓  {f}  ({kb:.0f} KB)')
        saved += 1
    else:
        print(f'  ✗  {f}  — not generated')
print()
print(f'  {saved}/{len(expected)} figures saved to: {OUT.resolve()}')
