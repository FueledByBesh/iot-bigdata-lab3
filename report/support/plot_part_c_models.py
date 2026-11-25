"""Plot the reproducible five-controller Part C replay."""
from pathlib import Path
import json
import os
BASE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(BASE / 'mpl_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = json.loads((BASE / 'part_c_model_comparison_results.json').read_text())
c = {x['name']: x for x in p['controllers']}
names = ['Fixed', 'Synthetic - naive', 'METR - naive', 'Synthetic - coordinated', 'METR - coordinated']
colors = ['#6b7280', '#0f766e', '#c2410c', '#0f766e', '#c2410c']
styles = ['-', '-', '-', '--', '--']
labels = ['Fixed', 'Synthetic / naive', 'METR / naive', 'Synthetic / coord', 'METR / coord']
fig, ax = plt.subplots(2, 2, figsize=(12, 6.8), sharex=True)
for name, color, style, label in zip(names, colors, styles, labels):
    v = c[name]
    x = np.asarray(v['hours'])
    q = np.asarray(v['queued_total'])
    assert len(x) % 60 == 0
    xq = x.reshape(-1,60).mean(axis=1)
    q = q.reshape(-1,60).mean(axis=1)
    w = np.asarray(v['wait_mean_5min'], dtype=float)
    xw = np.asarray(v['wait_bin_centres_hours'])
    for col in (0,1):
        if col == 1 and name == 'Fixed':
            continue
        ax[0,col].plot(xq,q,color=color,ls=style,lw=1.5,label=label)
        ax[1,col].plot(xw,w,color=color,ls=style,lw=1.5)
for a in ax.flat:
    a.grid(True, color='#e2e8f0', linewidth=.7)
    a.spines[['top','right']].set_visible(False)
    a.set_xlim(6,11)
    a.set_xticks(range(6,12),[f'{h:02d}:00' for h in range(6,12)])
    a.tick_params(labelsize=9)
ax[0,0].set_title('All controllers', fontsize=12, weight='bold')
ax[0,1].set_title('FNN controllers, enlarged scale', fontsize=12, weight='bold')
ax[0,0].set_ylabel('Queued vehicles\n(including entry buffers)', fontsize=10)
ax[1,0].set_ylabel('Mean passage wait (s)\nper 5-minute interval', fontsize=10)
for a in ax[1,:]:
    a.set_xlabel('Time on 13 March 2012', fontsize=10)
ax[0,0].set_ylim(bottom=0); ax[0,1].set_ylim(bottom=0)
ax[1,0].set_ylim(bottom=0); ax[1,1].set_ylim(bottom=0)
fig.legend(*ax[0,0].get_legend_handles_labels(), loc='upper center', ncol=5, frameon=False, fontsize=9)
fig.tight_layout(rect=(0,0,1,.93))
output = BASE.parent / 'figures/part_c_model_comparison.png'
fig.savefig(output,dpi=200,bbox_inches='tight')
print(output)
