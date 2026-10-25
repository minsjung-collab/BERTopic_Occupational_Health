"""Figures 2-4 (domain trajectories, period composition, journal heatmap) from domain_tables.py output.
Usage: python -I figs_domains.py domain_json out_dir"""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = json.load(open(sys.argv[1])); out = sys.argv[2]
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 11, "hatch.linewidth": 0.8})
ALPHA = sorted(R["S2"])  # alphabetical order, as in the original figures
COL = dict(zip(ALPHA, plt.cm.tab10.colors[:7]))
HATCH = dict(zip(ALPHA, ["", "/", ".", "x", "\\", "-", "o"]))
PER = ["Pre-COVID (2016-2019)", "COVID-19 (2020-2022)", "Post-COVID (2023-2025)"]
PLAB = ["Pre-COVID-19\n(2016–2019)", "COVID-19\n(2020–2022)", "Post-COVID-19\n(2023–2025)"]

# Fig 2: annual trajectories of the five largest domains
top5 = sorted(R["S2"], key=lambda k: -R["S2"][k])[:5]
years = sorted(int(y) for y in R["annual"])
fig, ax = plt.subplots(figsize=(11, 6.2))
for dm in top5:
    ax.plot(years, [R["annual"][str(y)][dm] for y in years], marker="o", lw=2, color=COL[dm], label=dm)
for x in (2020, 2023): ax.axvline(x, ls="--", lw=1.4, color="#1f77b4")
ax.set_xlabel("Year"); ax.set_ylabel("Percentage of substantive publications (%)")
ax.set_xticks(years); ax.grid(alpha=0.3); ax.set_ylim(0, 38)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=10)
fig.tight_layout(); fig.savefig(f"{out}/Fig2.png", dpi=600); plt.close(fig)

# Fig 3: period composition, stacked
fig, ax = plt.subplots(figsize=(11, 5.6))
bottom = np.zeros(3)
for dm in ALPHA:
    vals = np.array([100 * R["S3"][p][dm] / sum(R["S3"][p].values()) for p in PER])
    ax.bar(range(3), vals, 0.5, bottom=bottom, color=COL[dm], hatch=HATCH[dm], edgecolor="white", lw=0.8, label=dm)
    for i, v in enumerate(vals):
        if v >= 4: ax.text(i, bottom[i] + v / 2, f"{v:.1f}", ha="center", va="center", color="white", fontsize=10, fontweight="bold",
                           bbox=dict(boxstyle="round,pad=0.15", fc=(0, 0, 0, 0.35), ec="none"))
    bottom += vals
ax.set_xticks(range(3)); ax.set_xticklabels(PLAB); ax.set_xlabel("Historical period")
ax.set_ylabel("Percentage of substantive publications (%)"); ax.set_ylim(0, 100)
for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=10)
fig.tight_layout(); fig.savefig(f"{out}/Fig3.png", dpi=600); plt.close(fig)

# Fig 4: journal x domain heatmap
J = ["JOH", "OEM", "OM", "SJWEH"]
M = np.array([[100 * R["S5"][j][dm] / sum(R["S5"][j].values()) for dm in ALPHA] for j in J])
fig, ax = plt.subplots(figsize=(11, 5.2))
im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0)
for i in range(len(J)):
    for k in range(len(ALPHA)):
        ax.text(k, i, f"{M[i, k]:.1f}", ha="center", va="center", fontsize=10, color="black" if M[i, k] > 30 else "white")
wrap = {d: d.replace(", and ", ",\nand ").replace("Diseases and", "Diseases and\n") for d in ALPHA}
import textwrap
ax.set_xticks(range(len(ALPHA))); ax.set_xticklabels([textwrap.fill(d, 16) for d in ALPHA], fontsize=9)
ax.set_yticks(range(len(J))); ax.set_yticklabels(J); ax.set_xlabel("Thematic domain"); ax.set_ylabel("Journal")
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("Percentage of substantive publications (%)", fontsize=9)
fig.tight_layout(); fig.savefig(f"{out}/Fig4.png", dpi=600); plt.close(fig)
print("figs written")
