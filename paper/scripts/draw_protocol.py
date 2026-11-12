#!/usr/bin/env python3
"""Figure 0 of the paper: the protocol diagram (no data). `--no-readout` draws the fallback variant without the
Gemma 4 readout; `--v7` draws the v7 chain (two retrievers, oracle repair, patch and tests)."""
import sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

V7 = '--v7' in sys.argv
NO_READOUT = '--no-readout' in sys.argv
TOP = ['Issue text\n+ pre-fix snapshot', 'BM25 over\nmapped definitions\n→ 5 lexical seeds',
       'One-hop expansion\nover call graph', 'Ranked files\n→ top five', 'Gemma 4 31B\nfirst choice of file']
BELOW = {2: ('Clean graph, or\ndeletion / rewiring\n10–40%, seeds 13·37·101', '#f6dede'),
         4: ('Controls: repeated clean,\nreversed order, issue only,\nmatched placebo', '#e3f1e3')}
if V7:
    TOP = TOP[:2] + ['One-hop or PPR\nover call graph', 'Ranked files\n→ top five',
                     'Gemma 4 31B:\nfirst choice of file,\nor a unified diff', 'Diff applied;\nF2P + P2P tests\nin Docker → resolved?']
    BELOW = {2: ('Clean graph, deletion /\nrewiring 10–40%, or\noracle repair (traced calls)', '#f6dede'),
             4: ('Controls: repeated clean,\nreversed order, issue only,\nmatched placebo', '#e3f1e3')}
if NO_READOUT:
    TOP, BELOW = TOP[:4], {2: BELOW[2]}
n = len(TOP)
step, w = 0.98 / n, 0.98 / n - 0.035
xs = [0.01 + i * step for i in range(n)]
fig, ax = plt.subplots(figsize=(10 if n <= 5 else 11, 3.2)); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fs = 8.5 if n <= 5 else 7.8
for i, text in enumerate(TOP):
    colour = '#fdebd3' if 'Gemma' in text else '#e3e9f7' if 'Docker' in text else '#e8eef7'
    ax.add_patch(FancyBboxPatch((xs[i], 0.55), w, 0.32, boxstyle='round,pad=0.01', fc=colour, ec='#555'))
    ax.text(xs[i] + w / 2, 0.71, text, ha='center', va='center', fontsize=fs)
arrow = dict(arrowstyle='->', color='#333', lw=1.2)
for i in range(n - 1):
    ax.annotate('', (xs[i + 1], 0.71), (xs[i] + w + 0.002, 0.71), arrowprops=arrow)
for i, (text, colour) in BELOW.items():
    ax.add_patch(FancyBboxPatch((xs[i], 0.12), w, 0.32, boxstyle='round,pad=0.01', fc=colour, ec='#555'))
    ax.text(xs[i] + w / 2, 0.28, text, ha='center', va='center', fontsize=fs)
    ax.annotate('', (xs[i] + w / 2, 0.55), (xs[i] + w / 2, 0.45), arrowprops=arrow)
ax.text(0.30, 0.98, 'issue, snapshot, seeds and candidate pool fixed; only call edges change',
        ha='center', va='top', fontsize=8, style='italic')
out = 'f0_chain' if V7 else 'f0_protocol_no_readout' if NO_READOUT else 'f0_protocol'
plt.tight_layout(); plt.savefig(f'paper/figures/{out}.png', dpi=200)
