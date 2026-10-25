"""Recompute domain-level results (Tables S2-S7) from document-level BERTopic assignments.
Usage: python scripts/domain_tables.py data/document_topic_assignments.csv outputs/domain_corrected.json --corrected
--corrected applies the five topic-to-domain corrections (T19, T21, T24, T42, T48).
"""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency, norm

ODC = "Occupational Diseases, Cancer, and Hazardous Exposures"; PSY = "Psychosocial and Mental Health"
MSK = "Musculoskeletal, Physical, and Sensory Health"; WOES = "Work Organization, Employment, and Safety"
INF = "Infectious Diseases and Pandemic Health"; PRAC = "Occupational Health Practice, Prevention, and Methods"
WPR = "Worker Populations and Reproductive Health"
DOMS = [ODC, PSY, MSK, WOES, INF, PRAC, WPR]
CORRECTIONS = {19: ODC, 21: PRAC, 24: WOES, 42: ODC, 48: ODC}
PER = ["Pre-COVID (2016-2019)", "COVID-19 (2020-2022)", "Post-COVID (2023-2025)"]
J = ["JOH", "OEM", "OM", "SJWEH"]

d = pd.read_csv(sys.argv[1]); out = sys.argv[2]; corrected = "--corrected" in sys.argv
d["Dom"] = d["Domain_original"] if "Domain_original" in d.columns else d["Domain"]
if corrected:
    for t, v in CORRECTIONS.items(): d.loc[d.Topic == t, "Dom"] = v
s = d[d.Topic >= 0]
rng = np.random.default_rng(2026)

def cramer(tab):
    tab = np.asarray(tab, float); c, p, df, _ = chi2_contingency(tab); n = tab.sum(); r, k = tab.shape
    v = np.sqrt(c / (n * (min(r, k) - 1)))
    phi2 = c / n; phi2c = max(0, phi2 - (k - 1) * (r - 1) / (n - 1)); rc = r - (r - 1) ** 2 / (n - 1); kc = k - (k - 1) ** 2 / (n - 1)
    vbc = np.sqrt(phi2c / min(kc - 1, rc - 1))
    return dict(chi2=float(c), df=int(df), p=float(p), v=float(v), v_bc=float(vbc))

def boot_ci(tab, B=5000):
    tab = np.asarray(tab, float); n = int(tab.sum()); pr = (tab / tab.sum()).ravel(); vs = []
    for _ in range(B):
        t = rng.multinomial(n, pr).reshape(tab.shape)
        t = t[t.sum(1) > 0][:, t.sum(0) > 0]
        c = chi2_contingency(t)[0]; vs.append(np.sqrt(c / (n * (min(t.shape) - 1))))
    return [float(np.percentile(vs, 2.5)), float(np.percentile(vs, 97.5))]

def ct(df, row, col, rows, cols):
    return pd.crosstab(df[row], df[col]).reindex(index=rows, columns=cols, fill_value=0)

R = dict(corrected=corrected)
R["S2"] = {k: int(v) for k, v in s.Dom.value_counts().reindex(DOMS).items()}
tp = ct(s, "HistoricalPeriod", "Dom", PER, DOMS); tj = ct(s, "Journal", "Dom", J, DOMS)
R["S3"] = tp.to_dict(orient="index"); R["S5"] = tj.to_dict(orient="index")
R["period"] = cramer(tp); R["journal"] = cramer(tj)
R["period"]["ci"] = boot_ci(tp); R["journal"]["ci"] = boot_ci(tj)
# S4 pairwise z tests with BH-FDR across 21 comparisons
rows = []
for a, b in [(0, 1), (1, 2), (0, 2)]:
    A, Bp = tp.iloc[a], tp.iloc[b]; nA, nB = A.sum(), Bp.sum()
    for dm in DOMS:
        x1, x2 = A[dm], Bp[dm]; p = (x1 + x2) / (nA + nB); z = (x2 / nB - x1 / nA) / np.sqrt(p * (1 - p) * (1 / nA + 1 / nB))
        rows.append(dict(comp=f"{PER[a]} vs {PER[b]}", dom=dm, a=int(x1), na=int(nA), b=int(x2), nb=int(nB), z=float(z), p=float(2 * (1 - norm.cdf(abs(z))))))
ps = np.array([r["p"] for r in rows]); o = np.argsort(ps); m = len(ps); q = np.empty(m)
q[o] = np.minimum.accumulate((ps[o] * m / np.arange(1, m + 1))[::-1])[::-1]; q = np.minimum(q, 1)
for r, qq in zip(rows, q): r["p_fdr"] = float(qq)
R["S4"] = rows
# S6 COVID/AI by domain
R["S6"] = {dm: dict(total=int((s.Dom == dm).sum()), covid=int(s[s.Dom == dm].COVID19.sum()), ai=int(s[s.Dom == dm].AI_Related.sum())) for dm in DOMS}
R["S6"]["Outliers"] = dict(total=int((d.Topic == -1).sum()), covid=int(d[d.Topic == -1].COVID19.sum()), ai=int(d[d.Topic == -1].AI_Related.sum()))
# S7 sensitivity
d2 = d.copy(); d2.loc[d2.Topic == -1, "Dom"] = "Outlier"
tpo = ct(d2, "HistoricalPeriod", "Dom", PER, DOMS + ["Outlier"]); tjo = ct(d2, "Journal", "Dom", J, DOMS + ["Outlier"])
R["out_period"] = cramer(tpo); R["out_period"]["ci"] = boot_ci(tpo)
R["out_journal"] = cramer(tjo); R["out_journal"]["ci"] = boot_ci(tjo)
nx = s[s.Dom != INF]; D6 = [x for x in DOMS if x != INF]
tpx = ct(nx, "HistoricalPeriod", "Dom", PER, D6); tjx = ct(nx, "Journal", "Dom", J, D6)
R["ex_period"] = cramer(tpx); R["ex_period"]["ci"] = boot_ci(tpx)
R["ex_journal"] = cramer(tjx); R["ex_journal"]["ci"] = boot_ci(tjx)
R["ex_n"] = int(len(nx))
oc = (d.Topic == -1).astype(int)
R["outlier_period"] = cramer(pd.crosstab(d.HistoricalPeriod, oc).reindex(PER)); R["outlier_journal"] = cramer(pd.crosstab(d.Journal, oc).reindex(J))
R["outlier_pct_period"] = {p: float(100 * oc[d.HistoricalPeriod == p].mean()) for p in PER}
R["outlier_pct_journal"] = {j: float(100 * oc[d.Journal == j].mean()) for j in J}
# ODC share outside INF by period and pre vs COVID test
odc = {p: float(100 * (nx[nx.HistoricalPeriod == p].Dom == ODC).mean()) for p in PER}; R["odc_ex"] = odc
a = (nx[nx.HistoricalPeriod == PER[0]].Dom == ODC); b = (nx[nx.HistoricalPeriod == PER[1]].Dom == ODC)
pp = (a.sum() + b.sum()) / (len(a) + len(b)); z = (b.mean() - a.mean()) / np.sqrt(pp * (1 - pp) * (1 / len(a) + 1 / len(b)))
R["odc_ex_p"] = float(2 * (1 - norm.cdf(abs(z))))
# annual domain prevalence (for Fig 2) and year leader
ann = pd.crosstab(s.Year, s.Dom, normalize="index").reindex(columns=DOMS) * 100
R["annual"] = {int(y): {dm: float(ann.loc[y, dm]) for dm in DOMS} for y in ann.index}
R["odc_largest_years"] = [int(y) for y in ann.index if ann.loc[y].idxmax() == ODC]
json.dump(R, open(out, "w"), indent=1)
print("corrected" if corrected else "original", R["S2"])
for k in ["period", "journal", "out_period", "out_journal", "ex_period", "ex_journal", "outlier_period", "outlier_journal"]:
    v = R[k]; print(f"{k:14s} chi2={v['chi2']:.2f} df={v['df']} p={v['p']:.3g} V={v['v']:.3f} Vbc={v['v_bc']:.3f}", "CI" if "ci" in v else "", [round(x, 3) for x in v.get("ci", [])])
print("odc_ex", {k[:8]: round(v, 1) for k, v in odc.items()}, "p", round(R["odc_ex_p"], 3), "largest years", R["odc_largest_years"])
