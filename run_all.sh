#!/usr/bin/env bash
# Reproduce all reported domain-level and RQ4 results from the published topic assignments.
set -euo pipefail
python scripts/domain_tables.py data/document_topic_assignments.csv outputs/domain_original.json
python scripts/domain_tables.py data/document_topic_assignments.csv outputs/domain_corrected.json --corrected
python scripts/rq4_topics.py data/document_topic_assignments.csv data/keyword_flags.csv outputs
python scripts/figs_domains.py outputs/domain_corrected.json outputs/figures
python scripts/fig5_topics.py outputs/rq4_topic_stats.json outputs/figures/Fig5.png
