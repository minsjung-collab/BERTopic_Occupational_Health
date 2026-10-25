"""Benchmark correspondence analysis for the occupational health BERTopic manuscript.
Usage: python -I benchmarks.py corpus.pkl out_dir
Article-level lexical coverage of WHO/ILO Joint Estimates risk factor-outcome pairs and
ILO Global Strategy / SDG 8.8 system priorities; domain-level correspondence from published tables.
"""
import json, re, sys
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency

d = pd.read_pickle(sys.argv[1]); out = sys.argv[2]
t = d.CombinedText.fillna("").str.lower()
mesh = d.MeSH.fillna("").str.lower()
t = t.str.replace(r"international labou?r (organi[sz]ation|office)( \(ilo\))? (international )?classification|ilo (international )?classification", " ilo-radiograph-classification ", regex=True)
t = t.str.replace(r"(musculoskeletal|needle ?stick|sharps|kidney|lung|liver|renal|repetitive strain|overuse) injur(y|ies)", " ", regex=True)
d["period"] = np.where(d.Year <= 2019, "pre", np.where(d.Year <= 2022, "covid", "post"))
N = len(d)

# ---- WHO/ILO Joint Estimates 2016 (Pega et al. 2022, SJWEH 48:158-168, Table 1) ----
WHOILO = {  # group: (deaths, DALYs)
    "Injuries": (363283, 26440052),
    "Long working hours": (744924, 23258503),
    "Ergonomic factors": (0, 12267159),
    "Particulate matter, gases and fumes": (450381, 10855103),
    "Noise": (0, 8164140),
    "Carcinogens (13 agents)": (209481+7589+1452+165+452+1022+14728+743+7301+3881+42258+2564+25,
                                3973851+236361+85022+7181+17172+36059+470650+47199+229980+126900+1302917+91636+2343),
    "Asthmagens": (29641, 2104429),
}
tot_deaths = sum(v[0] for v in WHOILO.values()); tot_daly = sum(v[1] for v in WHOILO.values())

# exposure term (risk factor) and outcome term (WHO/ILO outcome) for each group
RF = {
    "Injuries": (r"(occupational|work[- ]?related|workplace|work|industrial|construction|farm|agricultural|traumatic|fatal|non-?fatal|lost[- ]time|recordable|serious|acute) injur(y|ies)|injur(y|ies) (at work|rates?|risk|claims?|surveillance|prevention)|(occupational|work[- ]?related|workplace|industrial|work|fatal) accidents?|accidents? at work|occupational fatalit|work[- ]?related (deaths?|fatalit)|fatal (work|occupational)|falls? from (height|ladders?|roofs?)|\\binjured workers?",
                 None),
    "Long working hours": (r"long (working|work) hours?|long hours|overtime|overwork|karoshi|extended working hours|weekly working hours|working hours per week|hours worked per week|working (time|hours) (and|regulation|directive|act)|[45][05] (or more )?hours",
                           r"cardiovascular|coronary|ischa?emic heart|heart disease|myocardial|\bstroke\b|cerebrovascular"),
    "Ergonomic factors": (r"ergonomic|manual (handling|lifting)|\blifting\b|awkward posture|physical(ly)? (work ?load|demanding|exposures?)|heavy (physical|lifting|work)|repetitive|trunk flexion|biomechanical",
                          r"back pain|neck pain|low[- ]back|lumbar|neck[/ ]shoulder|neck and shoulder|spinal|disc herniation|sciatica"),
    "Particulate matter, gases and fumes": (r"\bdusts?\b|\bfumes?\b|vapou?rs?|\bvgdf\b|particulate|\bgases\b|welding|silica|coal mine",
                                            r"\bcopd\b|chronic obstructive|airflow (obstruction|limitation)|emphysema|chronic bronchitis|lung function|\bfev1\b|spirometr"),
    "Noise": (r"\bnoise\b", r"hearing|deaf|tinnitus|audiometr|audiogram|\bnihl\b"),
    "Carcinogens (13 agents)": (r"asbestos|silica|arsenic|benzene|beryllium|cadmium|chromium|diesel|formaldehyde|\bnickel\b|polycyclic aromatic|\bpahs?\b|sulph?uric acid|trichloroethylene|\btce\b",
                                r"cancer|carcinom|mesothelioma|leuka?emia|neoplas|tumou?r|malignan|lymphoma"),
    "Asthmagens": (r"asthmagen|sensiti[sz]er|isocyanate|flour|latex|cleaning (agents|products)|disinfectant|work[- ]related asthma|occupational asthma",
                   r"asthma|wheez"),
}
# outcome-only pattern for injuries pair = exposure (injury is both exposure and outcome)

SYS = {  # ILO Global Strategy 2024-30 / SDG 8.8 system-level priorities
    "Injury surveillance, recording and compensation (SDG 8.8.1)":
        r"(?=.*\binjur)(?=.*(surveillance|registr|notif|record(ing|ed|s)\b|compensation|claims?\b|reporting))",
    "Migrant and foreign-born workers (SDG 8.8)": r"migrant workers?|immigrant workers?|foreign[- ](born )?workers?|migrant (farm|agricultural|seasonal)|labou?r migra|work(ers?)? migration|transients and migrants|emigrants and immigrants",
    "Precarious, informal and non-standard employment (SDG 8.8)":
        r"precarious|informal (sector|economy|work)|temporary (employment|work|agency)|contingent|gig |platform (work|economy)|non[- ]?standard employment|zero[- ]hours|self[- ]employ",
    "National OSH legislation, regulation and inspection (ILO pillar 1)":
        r"legislat|labou?r inspect|(osh|health and safety|safety and health|working environment|industrial safety|labou?r standards?) (act|law|legislation|regulations?|polic(y|ies))|occupational (health|safety) (law|polic(y|ies)|regulations?)|regulatory (standards?|limits?|agenc|framework|polic)|international labou?r (organi[sz]ation|office|standards?)|ilo conventions?|workers.? compensation (law|act|polic|system)",
    "Workplace OSH management systems (ILO pillar 3; ISO 45001)":
        r"(osh|safety|health and safety|safety and health) management( system)?|management systems?|iso 45001|ohsas|safety culture|(?<!psychosocial )safety climate|near[- ]miss",
    "Exposure limits, biomonitoring and industrial hygiene":
        r"exposure limit|\boels?\b|threshold limit|\btlvs?\b|biomonitor|biological monitoring|industrial hygien|occupational hygien",
    "Occupational health services": r"occupational health services?|occupational physicians?|occupational health nurs",
}

def has(p, src=None): return (t if src is None else src).str.contains(p, regex=True)

res = {"N": N, "tot_deaths": tot_deaths, "tot_daly": tot_daly, "rf": {}, "sys": {}}
flags = pd.DataFrame(index=d.index)
for g, (exp, outc) in RF.items():
    e = has(exp); pair = e & has(outc) if outc else e
    flags[g] = pair; flags[g + "|exp"] = e
    row = dict(deaths=WHOILO[g][0], daly=WHOILO[g][1], daly_pct=100*WHOILO[g][1]/tot_daly,
               death_pct=100*WHOILO[g][0]/tot_deaths, n_exp=int(e.sum()), n_pair=int(pair.sum()))
    for p in ["pre", "covid", "post"]:
        m = d.period == p; row[f"pair_{p}"] = int(pair[m].sum()); row[f"pct_{p}"] = 100*pair[m].mean()
    for j in ["JOH", "OEM", "OM", "SJWEH"]:
        m = d.Journal == j; row[f"pair_{j}"] = int(pair[m].sum()); row[f"pct_{j}"] = 100*pair[m].mean()
    res["rf"][g] = row
# shares among benchmark-matched mentions
tot_pair = sum(r["n_pair"] for r in res["rf"].values())
for g, r in res["rf"].items():
    r["mention_pct"] = 100*r["n_pair"]/tot_pair
    r["ratio"] = r["mention_pct"]/r["daly_pct"]
res["tot_pair_mentions"] = tot_pair
anyp = flags[list(RF)].any(axis=1)
res["any_pair_n"] = int(anyp.sum()); res["any_pair_pct"] = 100*anyp.mean()
res["any_pair_period"] = {p: 100*anyp[d.period == p].mean() for p in ["pre", "covid", "post"]}
res["any_pair_journal"] = {j: 100*anyp[d.Journal == j].mean() for j in ["JOH", "OEM", "OM", "SJWEH"]}
# dissimilarity (Duncan index) between mention distribution and DALY distribution, by period and journal
def duncan(mask):
    c = np.array([flags.loc[mask, g].sum() for g in RF]); s = c/c.sum()
    b = np.array([WHOILO[g][1] for g in RF]); b = b/b.sum()
    return 50*np.abs(s-b).sum()
res["duncan_all"] = duncan(d.index == d.index)
res["duncan_period"] = {p: duncan(d.period == p) for p in ["pre", "covid", "post"]}
res["duncan_journal"] = {j: duncan(d.Journal == j) for j in ["JOH", "OEM", "OM", "SJWEH"]}
# chi-square: risk-factor mention distribution by period
tab = np.array([[flags.loc[d.period == p, g].sum() for g in RF] for p in ["pre", "covid", "post"]])
c2, pv, df_, _ = chi2_contingency(tab); res["chi_period"] = dict(chi2=c2, df=df_, p=pv)
tabj = np.array([[flags.loc[d.Journal == j, g].sum() for g in RF] for j in ["JOH", "OEM", "OM", "SJWEH"]])
c2, pv, df_, _ = chi2_contingency(tabj); res["chi_journal"] = dict(chi2=c2, df=df_, p=pv)

for g, p in SYS.items():
    m = has(p) | (has(p, mesh) if "Migrant" in g else False); flags[g] = m
    row = dict(n=int(m.sum()), pct=100*m.mean())
    for q in ["pre", "covid", "post"]:
        row[f"n_{q}"] = int(m[d.period == q].sum()); row[f"pct_{q}"] = 100*m[d.period == q].mean()
    for j in ["JOH", "OEM", "OM", "SJWEH"]:
        row[f"pct_{j}"] = 100*m[d.Journal == j].mean()
    res["sys"][g] = row

# annual series for long working hours, injuries, ergonomic (for figure) and COVID check
res["annual"] = {g: [int(flags.loc[d.Year == y, g].sum()) for y in range(2016, 2026)] for g in RF}
res["annual_n"] = [int((d.Year == y).sum()) for y in range(2016, 2026)]
res["covid_check"] = int(has(r"covid|sars-cov-2|coronavirus|2019-ncov").sum())

# ---- domain-level correspondence (published Online Resource 1, Tables S2, S3, S5) ----
DOM = ["Occupational Diseases, Cancer, and Hazardous Exposures", "Psychosocial and Mental Health",
       "Musculoskeletal, Physical, and Sensory Health", "Work Organization, Employment, and Safety",
       "Infectious Diseases and Pandemic Health", "Occupational Health Practice, Prevention, and Methods",
       "Worker Populations and Reproductive Health"]
S2 = [741, 582, 425, 339, 232, 157, 75]
S3 = {"pre": [351, 253, 197, 140, 17, 75, 30], "covid": [210, 183, 124, 112, 120, 39, 22], "post": [180, 146, 104, 87, 95, 43, 23]}
S5 = {"JOH": [105, 155, 121, 35, 60, 63, 11], "OEM": [406, 123, 118, 121, 69, 13, 44],
      "OM": [151, 200, 102, 41, 80, 64, 6], "SJWEH": [79, 104, 84, 142, 23, 17, 14]}
D = lambda g: WHOILO[g][1]
alloc = {
    "primary": [D("Carcinogens (13 agents)")+D("Particulate matter, gases and fumes")+D("Asthmagens"), 0,
                D("Ergonomic factors")+D("Noise"), D("Injuries")+D("Long working hours"), 0, 0, 0],
    "alternative": [D("Carcinogens (13 agents)")+D("Particulate matter, gases and fumes")+D("Asthmagens"), D("Long working hours"),
                    D("Ergonomic factors")+D("Noise"), D("Injuries"), 0, 0, 0],
}
dom = {"domains": DOM, "research_pct": [100*x/sum(S2) for x in S2]}
for k, a in alloc.items():
    bp = [100*x/tot_daly for x in a]; dom[f"burden_pct_{k}"] = bp
    dom[f"ratio_{k}"] = [r/b if b > 0 else None for r, b in zip(dom["research_pct"], bp)]
    cp = [i for i, x in enumerate(a) if x > 0]
    dom[f"counterpart_pct_{k}"] = 100*sum(S2[i] for i in cp)/sum(S2)
    dom[f"counterpart_period_{k}"] = {p: 100*sum(v[i] for i in cp)/sum(v) for p, v in S3.items()}
    dom[f"counterpart_journal_{k}"] = {j: 100*sum(v[i] for i in cp)/sum(v) for j, v in S5.items()}
    # Duncan index between research distribution and burden distribution over 7 domains
    def dd(v): r = np.array(v)/sum(v); b = np.array(a)/sum(a); return 50*np.abs(r-b).sum()
    dom[f"duncan_{k}"] = dd(S2)
    dom[f"duncan_period_{k}"] = {p: dd(v) for p, v in S3.items()}
    dom[f"duncan_journal_{k}"] = {j: dd(v) for j, v in S5.items()}
res["domain"] = dom

json.dump(res, open(f"{out}/benchmark_stats.json", "w"), indent=1, default=float)
flags.insert(0, "PMID", d.PMID); flags.insert(1, "Journal", d.Journal); flags.insert(2, "Year", d.Year)
flags.to_csv(f"{out}/benchmark_flags.csv", index=False)
print(json.dumps(res, indent=1, default=lambda x: round(float(x), 2))[:9000])
