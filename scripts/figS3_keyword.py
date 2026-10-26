import json, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = json.load(open(sys.argv[1])); out = sys.argv[2]
rf = r["rf"]
tot_exp = sum(v["n_exp"] for v in rf.values())
order = sorted(rf, key=lambda g: rf[g]["daly_pct"])  # ascending -> largest at top
lab = {"Injuries": "Occupational injuries", "Long working hours": "Long working hours (≥55 h/week)",
       "Ergonomic factors": "Ergonomic factors", "Particulate matter, gases and fumes": "Particulate matter, gases and fumes",
       "Noise": "Noise", "Carcinogens (13 agents)": "Carcinogens (13 agents)", "Asthmagens": "Asthmagens"}
BUR, RES, INK, MUTED = "#C0702A", "#2A6FB0", "#222222", "#666666"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#999999",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})
fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 4.9), gridspec_kw={"width_ratios": [1.25, 1]}, sharey=True)
y = np.arange(len(order)); h = 0.36
dal = [rf[g]["daly_pct"] for g in order]; men = [rf[g]["mention_pct"] for g in order]
a.barh(y + h/2 + 0.02, dal, h, color=BUR, label="Share of WHO/ILO DALYs, 2016")
a.barh(y - h/2 - 0.02, men, h, color=RES, label="Share of keyword-matched publications")
for yi, v in zip(y, dal): a.text(v + 0.6, yi + h/2 + 0.02, f"{v:.1f}", va="center", fontsize=8, color=INK)
for yi, v in zip(y, men): a.text(v + 0.6, yi - h/2 - 0.02, f"{v:.1f}", va="center", fontsize=8, color=INK)
a.set_yticks(y); a.set_yticklabels([lab[g] for g in order])
a.set_xlabel("Percentage"); a.set_xlim(0, 35)
a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, frameon=False, fontsize=8)
a.set_title("A  Burden share vs research share", loc="left", fontsize=10, fontweight="bold", color=INK)
for s in ["top", "right"]: a.spines[s].set_visible(False)
a.grid(axis="x", color="#e6e6e6", lw=0.6); a.set_axisbelow(True)

pair = [rf[g]["ratio"] for g in order]
expo = [(100*rf[g]["n_exp"]/tot_exp)/rf[g]["daly_pct"] for g in order]
b.axvline(1, color="#888888", lw=1, ls="--", zorder=0)
for yi, p, e in zip(y, pair, expo):
    b.plot([min(p, e), max(p, e)], [yi, yi], color="#bbbbbb", lw=1.5, zorder=1)
b.scatter(pair, y, s=46, color=RES, zorder=3, label="Risk factor + WHO/ILO outcome (primary)")
b.scatter(expo, y, s=46, facecolors="white", edgecolors=RES, linewidths=1.6, zorder=3, label="Risk factor term only")
for yi, p in zip(y, pair): b.text(p, yi + 0.28, f"{p:.2f}", ha="center", fontsize=7.5, color=INK)
b.set_xscale("log"); b.set_xlim(0.07, 10)
b.set_xticks([0.1, 0.25, 0.5, 1, 2, 4, 8]); b.set_xticklabels(["0.1", "0.25", "0.5", "1", "2", "4", "8"])
b.set_xlabel("Attention-to-burden ratio (log scale)")
b.text(1.0, -0.95, "under-represented ←   → over-represented", fontsize=7.5, color=MUTED, ha="center")
b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, frameon=False, fontsize=8)
b.set_title("B  Ratio of research share to DALY share", loc="left", fontsize=10, fontweight="bold", color=INK)
for s in ["top", "right"]: b.spines[s].set_visible(False)
b.grid(axis="x", color="#e6e6e6", lw=0.6, which="major"); b.set_axisbelow(True)
b.set_ylim(-1.3, len(order) - 0.4)
fig.tight_layout()
fig.savefig(out, dpi=600)
