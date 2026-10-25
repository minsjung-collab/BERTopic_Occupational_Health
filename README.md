# BERTopic analysis of occupational health research, 2016–2025

Data, model, and code for the manuscript **"Research attention and the work-related burden of disease: a topic-modeling analysis of four occupational health journals, 2016–2025"** (Minsoo Jung, Dongduk Women's University).

The study applies BERTopic to 3190 PubMed-indexed publications from *Occupational and Environmental Medicine*, *Scandinavian Journal of Work, Environment & Health*, *Journal of Occupational Health*, and *Occupational Medicine* (2016–2025). It groups the 53 substantive topics into seven thematic domains and maps them to international occupational health benchmarks: the WHO/ILO Joint Estimates of the work-related burden of disease (2016), SDG target 8.8, and the ILO Global Strategy on Occupational Safety and Health 2024–2030.

## Contents

| Path | Description |
|---|---|
| `data/document_topic_assignments.csv` | One row per publication (N = 3190): PMID, DOI, title, journal, year, BERTopic topic (−1 = outlier), topic label, thematic domain, period, COVID-19 and AI keyword flags |
| `data/topic_information.csv`, `data/topic_top_words.csv` | Topic sizes and top terms from the baseline model |
| `data/topic_domain_indicator_crosswalk.csv` | Topic → domain → WHO/ILO risk factor group (primary and broad mapping) → SDG 8.8 / ILO Strategy priority |
| `data/whoilo_2016_burden.csv` | Deaths and DALYs attributable to the WHO/ILO risk factor groups, 2016 (Pega et al. 2022, *Scand J Work Environ Health* 48:158–168) |
| `data/keyword_flags.csv` | Article-level keyword matches used for validation (Supplementary Table S3) |
| `model/BERTopic_Model/` | Saved baseline BERTopic model (safetensors; embedding model `sentence-transformers/all-MiniLM-L6-v2`) |
| `scripts/domain_tables.py` | Domain-level results (Supplementary Tables S4–S9) |
| `scripts/rq4_topics.py` | Benchmark correspondence, attention-to-burden ratios, and 10-year trends (Tables 1–2; Supplementary Tables S2, S10) |
| `scripts/figs_domains.py`, `scripts/fig3_benchmark.py`, `scripts/figS3_keyword.py` | Figures 2–3 and Supplementary Figures S1–S3 |
| `scripts/keyword_benchmarks.py` | Keyword dictionaries (requires the abstract text; see below) |
| `scripts/original/temporal_analysis_original.py` | Original analysis script that produced the baseline outputs (Windows paths) |
| `outputs/` | Result files (JSON) and figures produced by the scripts |

## Reproducing the results

```bash
pip install -r requirements.txt
bash run_all.sh
```

`run_all.sh` regenerates every domain-level and benchmark result from `data/` and writes them to `outputs/`. The model check `json.load(open("model/BERTopic_Model/topics.json"))["topics"]` returns the same 3190 topic labels, in the same order, as `data/document_topic_assignments.csv`.

Abstract texts are not redistributed. They can be retrieved from PubMed with the PMIDs provided, using the query reported in the manuscript (`"[ISSN]"[ISSN] AND ("2016"[PDAT] : "2025"[PDAT]) AND hasabstract[text] AND english[lang]`, excluding editorials, comments, letters, news, biographies, errata, corrections, retractions, and expressions of concern; searched August 1, 2026).

## Topic-to-domain assignment

The `Domain` column is the assignment used in the manuscript. `Domain_original` records an earlier assignment in which five topics carried labels inconsistent with their content (T19, T21, T24, T42, T48); see `docs/domain_assignment.md`. `scripts/domain_tables.py` without `--corrected` reproduces results under the earlier assignment for comparison.

## Citation

See `CITATION.cff`. Code is released under the MIT License; the bibliographic metadata derive from PubMed (U.S. National Library of Medicine).
