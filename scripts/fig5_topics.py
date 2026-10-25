"""Figure 5: BERTopic research attention vs WHO/ILO burden. Usage: python -I fig5_topics.py rq4_topic_stats.json out.png"""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = json.load(open(sys.argv[1])); out = sys.argv[2]
P, B = R["primary"]["groups"], R["broad"]["groups"]
LAB = {"Occupational injuries": "Occupational injuries", "Long working hours": "Long working hours (≥55 h/week)",
       "Ergonomic factors": "Ergonomic factors (back and neck pain)", "Respiratory exposures (PMGF; asthmagens)": "Respiratory exposures (PMGF; asthmagens)",
       "Noise": "Noise (hearing loss)", "Carcinogens (13 agents)": "Carcinogens (13 agents)"}
BUR, RES, INK, MUTED = "#C0702A", "#2A6FB0", "#222222", "#666666"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#999999", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK})
order = sorted(P, key=lambda g: P[g]["daly_pct"])
fig = plt.figure(figsize=(10.4, 8.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], height_ratios=[1, 0.95], hspace=0.55, wspace=0.08)
a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1], sharey=a); c = fig.add_subplot(gs[1, :])
y = np.arange(len(order)); h = 0.36
dal = [P[g]["daly_pct"] for g in order]; res = [P[g]["share_mapped"] for g in order]
a.barh(y + h / 2 + 0.02, dal, h, color=BUR, label="Share of WHO/ILO DALYs, 2016")
a.barh(y - h / 2 - 0.02, res, h, color=RES, label="Share of publications in WHO/ILO-mapped BERTopic topics")
for yi, v in zip(y, dal): a.text(v + 0.6, yi + h / 2 + 0.02, f"{v:.1f}", va="center", fontsize=8, color=INK)
for yi, v, g in zip(y, res, order):
    a.text(v + 0.6, yi - h / 2 - 0.02, f"{v:.1f}" if v > 0 else "0 (no topic)", va="center", fontsize=8, color=INK)
a.set_yticks(y); a.set_yticklabels([LAB[g] for g in order]); a.set_xlim(0, 38); a.set_xlabel("Percentage")
a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, fontsize=8)
a.set_title("a  Burden share vs research share", loc="left", fontsize=10, fontweight="bold", color=INK)
for s in ["top", "right"]: a.spines[s].set_visible(False)
a.grid(axis="x", color="#e6e6e6", lw=0.6); a.set_axisbelow(True)

FLOOR = 0.07
b.axvline(1, color="#888888", lw=1, ls="--", zorder=0)
for yi, g in zip(y, order):
    p, q = P[g]["ratio"], B[g]["ratio"]
    pp, qq = max(p, FLOOR), max(q, FLOOR)
    b.plot([min(pp, qq), max(pp, qq)], [yi, yi], color="#bbbbbb", lw=1.5, zorder=1)
    if p > 0: b.scatter([p], [yi], s=46, color=RES, zorder=3); b.text(p, yi + 0.28, f"{p:.2f}", ha="center", fontsize=7.5, color=INK)
    else: b.scatter([FLOOR], [yi], s=46, marker="<", color=RES, zorder=3); b.text(FLOOR * 1.15, yi + 0.28, "0 (no topic)", ha="left", fontsize=7.5, color=INK)
    b.scatter([qq], [yi], s=46, facecolors="white", edgecolors=RES, linewidths=1.6, zorder=3)
b.scatter([], [], s=46, color=RES, label="Primary topic mapping"); b.scatter([], [], s=46, facecolors="white", edgecolors=RES, linewidths=1.6, label="Broad topic mapping")
b.set_xscale("log"); b.set_xlim(0.06, 6); b.set_xticks([0.1, 0.25, 0.5, 1, 2, 4]); b.set_xticklabels(["0.1", "0.25", "0.5", "1", "2", "4"])
b.set_xlabel("Attention-to-burden ratio (log scale)")
b.tick_params(axis="y", labelleft=False)
b.text(1.0, -0.9, "under-represented ←   → over-represented", fontsize=7.5, color=MUTED, ha="center")
b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, fontsize=8)
b.set_title("b  Research share ÷ DALY share", loc="left", fontsize=10, fontweight="bold", color=INK)
for s in ["top", "right"]: b.spines[s].set_visible(False)
b.grid(axis="x", color="#e6e6e6", lw=0.6); b.set_axisbelow(True); b.set_ylim(-1.2, len(order) - 0.4)

YEARS = list(range(2016, 2026))
COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]; MK = ["o", "s", "^", "D", "v"]
series = ["Respiratory exposures (PMGF; asthmagens)", "Ergonomic factors", "Carcinogens (13 agents)", "Occupational injuries", "Noise"]
for g, col, mk in zip(series, COLS, MK):
    v = P[g]["annual_pct"]; tr = P[g]["trend"]
    c.plot(YEARS, v, color=col, lw=2, marker=mk, ms=5, label=f"{LAB[g]}: OR {tr['or_per_year']:.2f}/yr, P = {tr['p']:.3f}".replace("P = 0.000", "P < 0.001"))
    c.text(2025.15, v[-1], LAB[g].split(" (")[0], va="center", fontsize=7.5, color=INK)
c.plot(YEARS, [0] * 10, color="#999999", lw=1, ls=":", label="Long working hours: no topic in any year")
c.set_xticks(YEARS); c.set_xlim(2015.6, 2027.0); c.set_ylim(0, 14); c.set_ylabel("Percentage of substantive publications")
c.set_xlabel("Publication year")
c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=7.5)
c.set_title("c  Annual share of substantive publications in WHO/ILO-mapped topics, 2016–2025", loc="left", fontsize=10, fontweight="bold", color=INK)
for s in ["top", "right"]: c.spines[s].set_visible(False)
c.grid(axis="y", color="#e6e6e6", lw=0.6); c.set_axisbelow(True)
fig.savefig(out, dpi=600, bbox_inches="tight")
