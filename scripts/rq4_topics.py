"""RQ4 topic-level benchmark correspondence using the ORIGINAL baseline BERTopic assignments.
Usage: python -I rq4_topics.py doc_level_csv keyword_flags_csv out_dir
"""
import json, sys
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency, spearmanr, norm

d = pd.read_csv(sys.argv[1]); kf = pd.read_csv(sys.argv[2]); out = sys.argv[3]
d = d.merge(kf, on="PMID", how="left", suffixes=("", "_kw"))
d["period"] = np.where(d.Year <= 2019, "pre", np.where(d.Year <= 2022, "covid", "post"))
YEARS = list(range(2016, 2026)); J = ["JOH", "OEM", "OM", "SJWEH"]
sub = d[d.Topic >= 0].copy()                       # substantive publications (n = 2551)

# ---- WHO/ILO Joint Estimates 2016 (Pega et al. 2022) ----
DALY = {"Occupational injuries": 26440052, "Long working hours": 23258503, "Ergonomic factors": 12267159,
        "Respiratory exposures (PMGF; asthmagens)": 10855103 + 2104429, "Noise": 8164140,
        "Carcinogens (13 agents)": 6627271}
DEATHS = {"Occupational injuries": 363283, "Long working hours": 744924, "Ergonomic factors": 0,
          "Respiratory exposures (PMGF; asthmagens)": 450381 + 29641, "Noise": 0, "Carcinogens (13 agents)": 291661}
TOT_DALY = 89716657; TOT_DEATHS = 1879890
# primary crosswalk: topic's defining content = the pair's risk factor and/or WHO/ILO outcome
PRIMARY = {"Occupational injuries": [3], "Long working hours": [], "Ergonomic factors": [1],
           "Respiratory exposures (PMGF; asthmagens)": [2, 27], "Noise": [29], "Carcinogens (13 agents)": [8, 20, 48]}
# broad crosswalk: adds topics on the same exposure with outcomes outside the WHO/ILO pair
BROAD = {"Occupational injuries": [3], "Long working hours": [40], "Ergonomic factors": [1, 22],
         "Respiratory exposures (PMGF; asthmagens)": [2, 27, 19], "Noise": [29, 49], "Carcinogens (13 agents)": [8, 20, 48, 4, 50]}
# system-level priorities (SDG 8.8; ILO Global Strategy 2024-30)
SYSTEM = {
 "Occupational injury and safety (SDG 8.8.1)": [3],
 "Psychosocial risks, violence and harassment, and mental health at work (ILO Strategy)": [9, 12, 13, 14, 16, 23, 30, 35, 38, 43, 45, 52],
 "Climate change and heat (ILO Strategy)": [37, 42],
 "Biological hazards and pandemics (ILO Strategy)": [0],
 "OSH services, surveillance, and exposure monitoring (ILO Strategy pillar 1)": [7, 25, 51],
 "Migrant workers (SDG 8.8)": [],
}

def logit_trend(y, x):
    X = sm.add_constant(np.asarray(x, float) - 2016)
    r = sm.GLM(np.asarray(y, float), X, family=sm.families.Binomial()).fit()
    b, se = r.params[1], r.bse[1]
    return dict(or_per_year=float(np.exp(b)), lo=float(np.exp(b - 1.96 * se)), hi=float(np.exp(b + 1.96 * se)), p=float(r.pvalues[1]))

def annual(mask_sub):
    return [100 * mask_sub[sub.Year == y].mean() for y in YEARS]

def summarize(topics, base=sub):
    m = base.Topic.isin(topics)
    row = dict(topics=topics, n=int(m.sum()), pct_sub=100 * m.mean(), pct_all=100 * d.Topic.isin(topics).mean())
    for p in ["pre", "covid", "post"]:
        mm = m[base.period == p]; row[f"n_{p}"] = int(mm.sum()); row[f"pct_{p}"] = 100 * mm.mean()
    for j in J:
        row[f"pct_{j}"] = 100 * m[base.Journal == j].mean()
    row["annual_pct"] = annual(m)
    row["annual_n"] = [int(m[base.Year == y].sum()) for y in YEARS]
    if m.sum() > 0:
        row["trend"] = logit_trend(m, base.Year)
        a, b = row["n_pre"], row["n_post"]; n1, n2 = (base.period == "pre").sum(), (base.period == "post").sum()
        pp = (a + b) / (n1 + n2); z = (b / n2 - a / n1) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
        row["p_pre_post"] = float(2 * (1 - norm.cdf(abs(z))))
    return row

R = {"n_sub": len(sub), "n_all": len(d), "annual_n_sub": [int((sub.Year == y).sum()) for y in YEARS]}
for name, cw in [("primary", PRIMARY), ("broad", BROAD)]:
    res = {g: summarize(t) for g, t in cw.items()}
    tot = sum(v["n"] for v in res.values())
    for g, v in res.items():
        v["daly_pct"] = 100 * DALY[g] / TOT_DALY; v["death_pct"] = 100 * DEATHS[g] / TOT_DEATHS
        v["share_mapped"] = 100 * v["n"] / tot if tot else 0
        v["ratio"] = v["share_mapped"] / v["daly_pct"]
        v["ratio_deaths"] = v["share_mapped"] / v["death_pct"] if v["death_pct"] > 0 else None
    # DALY share of the six groups relative to all 19 risk factors = 100% (the six groups are exhaustive)
    R[name] = dict(groups=res, mapped_n=tot, mapped_pct_sub=100 * tot / len(sub))
    alltopics = sorted({t for ts in cw.values() for t in ts})
    R[name]["any"] = summarize(alltopics)
    # dissimilarity index overall, by period, by year, by journal (mapped-topic distribution vs DALY distribution)
    def dis(mask):
        c = np.array([sub[mask].Topic.isin(t).sum() for t in cw.values()], float)
        s = c / c.sum(); b = np.array([DALY[g] for g in cw]) / TOT_DALY
        return float(50 * np.abs(s - b).sum())
    R[name]["dis_all"] = dis(sub.index == sub.index)
    R[name]["dis_period"] = {p: dis(sub.period == p) for p in ["pre", "covid", "post"]}
    R[name]["dis_year"] = [dis(sub.Year == y) for y in YEARS]
    rho, prho = spearmanr(YEARS, R[name]["dis_year"]); R[name]["dis_year_spearman"] = dict(rho=float(rho), p=float(prho))
    R[name]["dis_journal"] = {j: dis(sub.Journal == j) for j in J}
    tab = np.array([[sub[sub.period == p].Topic.isin(t).sum() for t in cw.values() if t] for p in ["pre", "covid", "post"]])
    c2, pv, df_, _ = chi2_contingency(tab); R[name]["chi_period"] = dict(chi2=float(c2), df=int(df_), p=float(pv))
    tabj = np.array([[sub[sub.Journal == j].Topic.isin(t).sum() for t in cw.values() if t] for j in J])
    c2, pv, df_, _ = chi2_contingency(tabj); R[name]["chi_journal"] = dict(chi2=float(c2), df=int(df_), p=float(pv))

R["system"] = {g: summarize(t) if t else dict(topics=[], n=0, pct_sub=0.0) for g, t in SYSTEM.items()}

# ---- convergent validity: where keyword-matched publications fall ----
KW = {"Occupational injuries": "Injuries", "Long working hours": "Long working hours", "Ergonomic factors": "Ergonomic factors",
      "Respiratory exposures (PMGF; asthmagens)": ["Particulate matter, gases and fumes", "Asthmagens"], "Noise": "Noise",
      "Carcinogens (13 agents)": "Carcinogens (13 agents)"}
val = {}
for g, k in KW.items():
    km = d[k].any(axis=1) if isinstance(k, list) else d[k].astype(bool)
    inmap = d.Topic.isin(PRIMARY[g]); outl = d.Topic == -1
    val[g] = dict(n_kw=int(km.sum()), pct_in_mapped=100 * (km & inmap).sum() / km.sum(), pct_outlier=100 * (km & outl).sum() / km.sum(),
                  pct_mapped_with_kw=100 * (km & inmap).sum() / inmap.sum() if inmap.sum() else None)
lwh_any = d["Long working hours|exp"].astype(bool)
val["Long working hours (risk factor term, any outcome)"] = dict(n_kw=int(lwh_any.sum()), pct_outlier=100 * (lwh_any & (d.Topic == -1)).sum() / lwh_any.sum(),
    top_topics={int(k): int(v) for k, v in d[lwh_any].Topic.value_counts().head(6).items()})
R["validity"] = val
json.dump(R, open(f"{out}/rq4_topic_stats.json", "w"), indent=1)
print(json.dumps({k: R[k] for k in ["n_sub"]}, indent=1))
for name in ["primary", "broad"]:
    print("==", name, "mapped", R[name]["mapped_n"], f"{R[name]['mapped_pct_sub']:.1f}%", "dis", round(R[name]["dis_all"], 1),
          "dis_period", {k: round(v, 1) for k, v in R[name]["dis_period"].items()}, "spearman", R[name]["dis_year_spearman"])
    print("   chi period", R[name]["chi_period"], "journal", R[name]["chi_journal"])
    print("   any trend", R[name]["any"]["trend"], [round(x, 1) for x in R[name]["any"]["annual_pct"]])
    for g, v in R[name]["groups"].items():
        tr = v.get("trend", {})
        print(f"   {g[:34]:34s} T{v['topics']} n={v['n']:3d} sub%={v['pct_sub']:4.1f} share={v['share_mapped']:4.1f} DALY={v['daly_pct']:4.1f} ratio={v['ratio']:.2f} "
              f"pre/cov/post={v['pct_pre']:.1f}/{v['pct_covid']:.1f}/{v['pct_post']:.1f} OR/yr={tr.get('or_per_year', float('nan')):.3f} p={tr.get('p', float('nan')):.3f}")
print("== system")
for g, v in R["system"].items():
    tr = v.get("trend", {})
    print(f"   {g[:60]:60s} n={v['n']:4d} {v['pct_sub']:4.1f}% " + (f"pre/cov/post={v['pct_pre']:.1f}/{v['pct_covid']:.1f}/{v['pct_post']:.1f} OR/yr={tr['or_per_year']:.3f} p={tr['p']:.3f}" if tr else ""))
print("== validity"); [print("  ", g, {k: (round(x, 1) if isinstance(x, float) else x) for k, x in v.items()}) for g, v in val.items()]
