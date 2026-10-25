#!/usr/bin/env bash
# Reproduce all reported domain-level and RQ4 results from the published topic assignments.
set -euo pipefail
python scripts/domain_tables.py data/document_topic_assignments.csv outputs/domain_original.json
python scripts/domain_tables.py data/document_topic_assignments.csv outputs/domain_corrected.json --corrected
python scripts/rq4_topics.py data/document_topic_assignments.csv data/keyword_flags.csv outputs
python scripts/figs_domains.py outputs/domain_corrected.json outputs/figures  # writes Fig2/Fig3/Fig4.png = Supplementary Figures S1, S2 and main Figure 2
python scripts/fig3_benchmark.py outputs/rq4_topic_stats.json outputs/figures/Figure3_benchmark_correspondence.png
