# Topic-to-domain assignment

The seven thematic domains aggregate the 53 substantive BERTopic topics. During preparation of the manuscript, five topics were found to carry domain labels inconsistent with their top terms and representative documents. They were reassigned as follows:

| Topic | Label | Earlier domain | Domain used |
|---|---|---|---|
| T19 | Nanoparticle and Experimental Pulmonary Toxicity | Psychosocial and Mental Health | Occupational Diseases, Cancer, and Hazardous Exposures |
| T21 | Workplace Psychosocial Scales and Measurement Validation | Occupational Diseases, Cancer, and Hazardous Exposures | Occupational Health Practice, Prevention, and Methods |
| T24 | Return to Work and Vocational Rehabilitation | Occupational Diseases, Cancer, and Hazardous Exposures | Work Organization, Employment, and Safety |
| T42 | Heat Stress and Kidney Injury in Agricultural Workers | Psychosocial and Mental Health | Occupational Diseases, Cancer, and Hazardous Exposures |
| T48 | Occupational Lung Cancer, Silica, and Job-Exposure Assessment | Work Organization, Employment, and Safety | Occupational Diseases, Cancer, and Hazardous Exposures |

Effect on the main results (`outputs/domain_original.json` vs `outputs/domain_corrected.json`): the period association was unchanged (Cramér's V = 0.161 in both), the journal association strengthened (V = 0.249 → 0.270), and the period association remained non-significant after excluding the infectious disease domain (P = 0.60 → 0.65). Topic assignments and the benchmark analysis (RQ4) do not depend on domain labels.
