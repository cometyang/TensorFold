"""Plot measured decode throughput, keeping the +20% target visible."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
summary = json.loads((root / 'summary.json').read_text())
contexts = ('2048', '65536', '1047040')
labels = ('2K', '64K', 'Near 1M\n1,047,040 tokens')
rows = [summary['contexts'][key] for key in contexts]
baseline = np.array([row['baseline_tps'] for row in rows])
candidate = np.array([row['candidate_tps'] for row in rows])
target = baseline * 1.2
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
x = np.arange(3)
ax.bar(x-.18, baseline, width=.34, color='#6c7d90', label='Bracketed baseline A1 / A2')
ax.bar(x+.18, candidate, width=.34, color='#258d98', label='Context-calibrated allocator')
for index, (before, after, goal, row) in enumerate(zip(baseline, candidate, target, rows)):
    ax.plot([index-.4, index+.4], [goal, goal], color='#c67d22', ls='--', lw=2,
            label='+20% throughput target' if index == 0 else None)
    ax.text(index, goal+2, f"{row['improvement_pct']:+.2f}% measured", ha='center', fontweight='bold')
    ax.text(index-.18, before-2, f'{before:.2f}', ha='center', va='top', color='white', fontsize=10)
    ax.text(index+.18, after-2, f'{after:.2f}', ha='center', va='top', color='white', fontsize=10)
ax.set_xticks(x, labels)
ax.set_ylabel('Decode tokens/s (higher is better)')
ax.set_ylim(0, max(target)+25)
ax.grid(axis='y', alpha=.15)
ax.set_axisbelow(True)
ax.legend(loc='upper left', frameon=False, fontsize=10)
fig.suptitle('TensorFold · context-calibrated speculative decode', fontsize=17, fontweight='bold')
ax.set_title('Two DGX Sparks · 2200 MHz ceiling · 1024 output tokens · three measured repeats', fontsize=10, pad=16)
fig.supxlabel('Geometric mean across fixture medians; A1/A2 bracket B. 2K and 64K: prose, code, retrieval.\n'
              'Near 1M: one retrieval fixture. Client post-first-token timing; no claim of statistical significance.',
              fontsize=9, color='#48525b')
fig.savefig(root/'decode-throughput.png', dpi=170)
fig.savefig(root/'decode-throughput.svg')
print(root/'decode-throughput.png')
