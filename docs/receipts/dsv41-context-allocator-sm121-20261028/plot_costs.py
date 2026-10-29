"""Plot measured relative costs without presenting them as throughput gains."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

profiles = json.loads(Path('costs.json').read_text())['profiles']
fig, ax = plt.subplots(figsize=(8.3, 5.1))
for profile, color, label in zip(profiles, ['#2563eb', '#059669', '#db2777'], ['2K', '64K', 'Near 1M']):
    costs = [value + profile['overhead_ms'] for value in profile['forward_ms']]
    ax.plot(range(1, 7), [value / costs[0] for value in costs], marker='o', color=color,
            label=label + ' measured')
legacy = [30, 35.2, 39.9, 44.9, 48.6, 52.1]
ax.plot(range(1, 7), [value / legacy[0] for value in legacy], ls='--', color='#374151',
        label='Legacy table, H = 0 scenario')
ax.set(xlabel='Verified rows per round (1 pending token + drafts)', ylabel='Round cost / one-row cost',
       title='Calibration changes relative verification costs only slightly', xticks=range(1, 7))
ax.grid(alpha=.18)
ax.legend(frameon=False)
fig.tight_layout(rect=[0, .14, 1, 1])
fig.text(.025, .025, 'Measured: forward + calibrated overhead. Two GB10s, 2200 MHz cap.\n'
         'Training cost curves explain choices; they are not end-to-end throughput measurements.',
         fontsize=9, color='#4b5563')
for suffix in ['png', 'svg']:
    fig.savefig('calibration-cost-curves.' + suffix, dpi=180)
