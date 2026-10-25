# ============================================================
# Occupational Health BERTopic
# Temporal Evolution, COVID-19, AI Emergence, and Journal Analysis
# 2016-2025
#
# Baseline model:
#   N = 3,190
#   Topics = 53
#   Outliers = 639 (20.03%)
#
# IMPORTANT:
# This script DOES NOT refit BERTopic.
# It uses the previously saved baseline model and corpus.
# ============================================================

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from bertopic import BERTopic


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = r"C:\Users\DDWU\Desktop\Occupational Health BERTopic"

CORPUS_FILE = os.path.join(
    BASE_DIR,
    "Occupational_Health_BERTopic_2016_2025.xlsx"
)

BASELINE_DIR = os.path.join(
    BASE_DIR,
    "BERTopic_Baseline_Results"
)

MODEL_DIR = os.path.join(
    BASELINE_DIR,
    "BERTopic_Model"
)

ASSIGNMENT_FILE = os.path.join(
    BASELINE_DIR,
    "04_Document_Topic_Assignments.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "BERTopic_Temporal_Analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. BASIC SETTINGS
# ============================================================

START_YEAR = 2016
END_YEAR = 2025

YEARS = list(range(START_YEAR, END_YEAR + 1))

JOURNALS = [
    "OEM",
    "SJWEH",
    "JOH",
    "OM"
]

N_TOPICS = 53

print("\n" + "=" * 70)
print("Occupational Health BERTopic")
print("Temporal Evolution and Historical Event Analysis")
print("2016-2025")
print("=" * 70)

print("\nCorpus:")
print(CORPUS_FILE)

print("\nBaseline model:")
print(MODEL_DIR)

print("\nOutput:")
print(OUTPUT_DIR)


# ============================================================
# 3. LOAD CORPUS
# ============================================================

print("\n" + "=" * 70)
print("STEP 1. Loading corpus")
print("=" * 70)

if not os.path.exists(CORPUS_FILE):
    raise FileNotFoundError(
        f"Corpus file not found:\n{CORPUS_FILE}"
    )

corpus_df = pd.read_excel(CORPUS_FILE)

print("Loaded records:", len(corpus_df))
print("Columns:")
print(list(corpus_df.columns))


# ============================================================
# 4. STANDARDIZE IMPORTANT COLUMN NAMES
# ============================================================

def find_column(df, candidates):

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    lower_map = {
        str(c).lower().strip(): c
        for c in df.columns
    }

    for candidate in candidates:
        key = candidate.lower().strip()

        if key in lower_map:
            return lower_map[key]

    return None


pmid_col = find_column(
    corpus_df,
    ["PMID", "Pmid", "pmid"]
)

journal_col = find_column(
    corpus_df,
    ["Journal", "journal"]
)

year_col = find_column(
    corpus_df,
    ["Year", "year", "PublicationYear"]
)

text_col = find_column(
    corpus_df,
    [
        "CombinedText",
        "combinedtext",
        "Combined Text",
        "Abstract",
        "abstract"
    ]
)

title_col = find_column(
    corpus_df,
    [
        "Title",
        "title",
        "ArticleTitle"
    ]
)

doi_col = find_column(
    corpus_df,
    [
        "DOI",
        "doi"
    ]
)

if journal_col is None:
    raise ValueError("Journal column could not be identified.")

if year_col is None:
    raise ValueError("Year column could not be identified.")

if text_col is None:
    raise ValueError("CombinedText/Abstract column could not be identified.")


print("\nDetected columns:")

print("PMID:", pmid_col)
print("Journal:", journal_col)
print("Year:", year_col)
print("Text:", text_col)
print("Title:", title_col)
print("DOI:", doi_col)


# ============================================================
# 5. CLEAN CORPUS
# ============================================================

corpus_df[year_col] = pd.to_numeric(
    corpus_df[year_col],
    errors="coerce"
)

corpus_df = corpus_df[
    corpus_df[year_col].between(
        START_YEAR,
        END_YEAR
    )
].copy()

corpus_df[journal_col] = (
    corpus_df[journal_col]
    .astype(str)
    .str.strip()
)

corpus_df[text_col] = (
    corpus_df[text_col]
    .fillna("")
    .astype(str)
)

corpus_df["Year"] = corpus_df[year_col].astype(int)

corpus_df["Journal"] = corpus_df[journal_col]

corpus_df["CombinedText"] = corpus_df[text_col]


# ============================================================
# 6. LOAD DOCUMENT-TOPIC ASSIGNMENTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 2. Loading baseline topic assignments")
print("=" * 70)

if not os.path.exists(ASSIGNMENT_FILE):

    raise FileNotFoundError(
        f"Topic assignment file not found:\n{ASSIGNMENT_FILE}"
    )

assignments = pd.read_csv(
    ASSIGNMENT_FILE
)

print(
    "Loaded topic assignments:",
    len(assignments)
)

print(
    "Assignment columns:"
)

print(
    list(assignments.columns)
)


# ============================================================
# 7. IDENTIFY TOPIC COLUMN
# ============================================================

topic_col = find_column(
    assignments,
    [
        "Topic",
        "topic",
        "Topics",
        "topics"
    ]
)

if topic_col is None:
    raise ValueError(
        "Topic column could not be identified in "
        "04_Document_Topic_Assignments.csv"
    )

print(
    "Detected topic column:",
    topic_col
)


# ============================================================
# 8. MERGE ASSIGNMENTS WITH CORPUS
# ============================================================

print("\nMerging corpus and topic assignments...")


# Preferred merge: PMID
if pmid_col is not None:

    assignment_pmid_col = find_column(
        assignments,
        ["PMID", "Pmid", "pmid"]
    )

else:

    assignment_pmid_col = None


if (
    pmid_col is not None
    and assignment_pmid_col is not None
):

    corpus_df["PMID_merge"] = (
        corpus_df[pmid_col]
        .astype(str)
        .str.strip()
    )

    assignments["PMID_merge"] = (
        assignments[assignment_pmid_col]
        .astype(str)
        .str.strip()
    )

    analysis_df = corpus_df.merge(
        assignments[
            [
                "PMID_merge",
                topic_col
            ]
        ],
        on="PMID_merge",
        how="left"
    )

else:

    print(
        "PMID merge unavailable. "
        "Using row-order matching."
    )

    if len(corpus_df) != len(assignments):

        raise ValueError(
            "Corpus and topic-assignment row counts differ; "
            "PMID matching is unavailable."
        )

    analysis_df = corpus_df.copy()

    analysis_df[topic_col] = assignments[
        topic_col
    ].values


analysis_df = analysis_df.rename(
    columns={
        topic_col: "Topic"
    }
)


# ============================================================
# 9. QUALITY CONTROL
# ============================================================

analysis_df["Topic"] = pd.to_numeric(
    analysis_df["Topic"],
    errors="coerce"
)

missing_topics = analysis_df["Topic"].isna().sum()

print("\nMissing topic assignments:", missing_topics)

if missing_topics > 0:

    print(
        "WARNING:",
        missing_topics,
        "documents do not have topic assignments."
    )

analysis_df = analysis_df[
    analysis_df["Topic"].notna()
].copy()

analysis_df["Topic"] = analysis_df["Topic"].astype(int)

analysis_df["TopicLabel"] = analysis_df["Topic"]


print(
    "\nDocuments entering temporal analysis:",
    len(analysis_df)
)

print(
    "\nTopic distribution:"
)

print(
    analysis_df["Topic"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 10. LOAD BERTopic MODEL
# ============================================================

print("\n" + "=" * 70)
print("STEP 3. Loading baseline BERTopic model")
print("=" * 70)

if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(
        f"BERTopic model directory not found:\n{MODEL_DIR}"
    )

print("Loading model...")

try:

    topic_model = BERTopic.load(
        MODEL_DIR,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )

except TypeError:

    print(
        "Standard load failed. "
        "Trying default BERTopic.load()."
    )

    topic_model = BERTopic.load(
        MODEL_DIR
    )

print("BERTopic model loaded successfully.")


# ============================================================
# 11. TOPIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 4. Extracting topic information")
print("=" * 70)

topic_info = topic_model.get_topic_info()

topic_info = topic_info.copy()

topic_info = topic_info.rename(
    columns={
        "Topic": "Topic",
        "Count": "TopicCount",
        "Name": "BERTopicName"
    }
)

print(
    "\nNumber of topics including outlier:",
    len(topic_info)
)

print(
    "\nTopic information:"
)

print(
    topic_info[
        [
            "Topic",
            "TopicCount",
            "BERTopicName"
        ]
    ].to_string(index=False)
)


# ============================================================
# 12. SUBSTANTIVE TOPIC LABELS
# ============================================================

SUBSTANTIVE_LABELS = {

    0: "COVID-19 and Workplace Health",

    1: "Occupational Musculoskeletal Disorders and Pain",

    2: "Occupational Respiratory Disease and Inhalational Exposures",

    3: "Occupational Injuries, Safety, and Precarious Employment",

    4: "Occupational Cancer and Chemical Exposures",

    5: "Sleep, Shift Work, and Fatigue",

    6: "Retirement, Disability, and Labor-Market Exit",

    7: "Occupational Health Practice, Physicians, and Training",

    8: "Asbestos Exposure, Mesothelioma, and Lung Cancer",

    9: "Workplace Mental Health Interventions",

    10: "Sickness Absence and Work Disability",

    11: "Occupational Physical Activity and Sedentary Behavior",

    12: "Occupational Burnout and Emotional Exhaustion",

    13: "Military Service, Veterans, and Occupational Mental Health",

    14: "Work Engagement, Job Demands, and Psychological Distress",

    15: "Pregnancy, Maternal Occupational Exposures, and Birth Outcomes",

    16: "Workplace Violence, Harassment, and Gender-Based Violence",

    17: "Hand-Arm Vibration Syndrome",

    18: "Occupational Radiation Exposure and Cancer",

    19: "Nanoparticle and Experimental Pulmonary Toxicity",

    20: "Respirable Crystalline Silica Exposure",

    21: "Workplace Psychosocial Scales and Measurement Validation",

    22: "Carpal Tunnel Syndrome and Biomechanical Exposures",

    23: "Suicide and Suicidal Behavior in Working Populations",

    24: "Return to Work and Vocational Rehabilitation",

    25: "Occupational Exposure Monitoring and Industrial Hygiene",

    26: "Occupational Chemical Exposures, Pregnancy, and Fetal Growth",

    27: "Coal Mine Dust Lung Disease and Pneumoconiosis",

    28: "Night Shift Work and Cancer Risk",

    29: "Occupational Noise and Hearing Loss",

    30: "Job Insecurity, Work-Family Conflict, and Mental Health",

    31: "Workplace Physical-Activity Interventions",

    32: "Occupational Lead Exposure and Health Outcomes",

    33: "Occupational Skin Disease and Contact Dermatitis",

    34: "Firefighter Occupational Cancer and Mortality",

    35: "Police, First Responders, PTSD, and Occupational Stress",

    36: "Ambient Air Pollution and Respiratory/Cardiopulmonary Health",

    37: "Heat Exposure, Temperature, and Health",

    38: "Work Stress and Metabolic Disease/Diabetes",

    39: "Fitness and Physical Performance in First Responders",

    40: "Night/Shift Work and Cardiovascular/Cerebrovascular Health",

    41: "Occupational Pesticide Exposure and Health",

    42: "Heat Stress and Kidney Injury in Agricultural Workers",

    43: "Young Workers, Employment, and Mental Health",

    44: "Cancer Survivorship, Fatigue, and Work Ability",

    45: "Firefighter PTSD and Traumatic Stress",

    46: "Extremely Low-Frequency Magnetic Fields and Cancer",

    47: "Seafarer Occupational Health and Mortality",

    48: "Occupational Lung Cancer, Silica, and Job-Exposure Assessment",

    49: "Noise Exposure and Cardiovascular Health",

    50: "Welding Fumes, Metal Exposure, and Cancer",

    51: "Occupational Coding, Job Histories, and Job-Exposure Matrices",

    52: "Workplace Bullying and Suicidal/Mental Health Outcomes"
}


analysis_df["SubstantiveTopic"] = (
    analysis_df["Topic"]
    .map(SUBSTANTIVE_LABELS)
)

analysis_df["SubstantiveTopic"] = (
    analysis_df["SubstantiveTopic"]
    .fillna(
        analysis_df["Topic"].apply(
            lambda x: (
                "Outlier"
                if x == -1
                else f"Topic {x}"
            )
        )
    )
)


# ============================================================
# 13. TOPIC SUMMARY TABLE
# ============================================================

topic_summary = (
    analysis_df
    .groupby(
        [
            "Topic",
            "SubstantiveTopic"
        ]
    )
    .size()
    .reset_index(
        name="Documents"
    )
)

topic_summary["Percentage"] = (
    topic_summary["Documents"]
    / len(analysis_df)
    * 100
)

topic_summary = topic_summary.sort_values(
    "Topic"
)

topic_summary.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "01_Topic_Summary.xlsx"
    ),
    index=False
)


# ============================================================
# 14. ANNUAL TOPIC PREVALENCE
# ============================================================

print("\n" + "=" * 70)
print("STEP 5. Calculating annual topic prevalence")
print("=" * 70)


annual_counts = pd.crosstab(
    analysis_df["Year"],
    analysis_df["Topic"]
)

annual_counts = annual_counts.reindex(
    YEARS,
    fill_value=0
)

# Keep all topics 0-52
all_topics = list(range(53))

annual_counts = annual_counts.reindex(
    columns=all_topics,
    fill_value=0
)

annual_percent = (
    annual_counts
    .div(
        annual_counts.sum(axis=1),
        axis=0
    )
    * 100
)

annual_counts_long = (
    annual_counts
    .reset_index()
    .melt(
        id_vars="Year",
        var_name="Topic",
        value_name="Documents"
    )
)

annual_percent_long = (
    annual_percent
    .reset_index()
    .melt(
        id_vars="Year",
        var_name="Topic",
        value_name="Percentage"
    )
)

annual_long = annual_counts_long.merge(
    annual_percent_long,
    on=["Year", "Topic"]
)

annual_long["Topic"] = (
    annual_long["Topic"].astype(int)
)

annual_long["SubstantiveTopic"] = (
    annual_long["Topic"]
    .map(SUBSTANTIVE_LABELS)
)

annual_long["SubstantiveTopic"] = (
    annual_long["SubstantiveTopic"]
    .fillna(
        annual_long["Topic"].apply(
            lambda x: f"Topic {x}"
        )
    )
)

annual_long.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "02_Annual_Topic_Prevalence.xlsx"
    ),
    index=False
)


# ============================================================
# 15. PERIOD-BASED PREVALENCE
# ============================================================

print("\n" + "=" * 70)
print("STEP 6. Calculating historical-period prevalence")
print("=" * 70)


def classify_period(year):

    if year <= 2019:
        return "Pre-COVID (2016-2019)"

    elif year <= 2021:
        return "COVID-19 Acute Period (2020-2021)"

    else:
        return "Post-Acute COVID Period (2022-2025)"


analysis_df["HistoricalPeriod"] = (
    analysis_df["Year"]
    .apply(classify_period)
)


period_order = [
    "Pre-COVID (2016-2019)",
    "COVID-19 Acute Period (2020-2021)",
    "Post-Acute COVID Period (2022-2025)"
]


period_counts = pd.crosstab(
    analysis_df["HistoricalPeriod"],
    analysis_df["Topic"]
)

period_counts = period_counts.reindex(
    period_order,
    fill_value=0
)

period_counts = period_counts.reindex(
    columns=all_topics,
    fill_value=0
)

period_percent = (
    period_counts
    .div(
        period_counts.sum(axis=1),
        axis=0
    )
    * 100
)


period_long = (
    period_percent
    .reset_index()
    .melt(
        id_vars="HistoricalPeriod",
        var_name="Topic",
        value_name="Percentage"
    )
)

period_long["Topic"] = (
    period_long["Topic"].astype(int)
)

period_long["SubstantiveTopic"] = (
    period_long["Topic"]
    .map(SUBSTANTIVE_LABELS)
)

period_long.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "03_Historical_Period_Prevalence.xlsx"
    ),
    index=False
)


# ============================================================
# 16. TOPIC CHANGE FROM PRE-COVID TO POST-COVID
# ============================================================

period_wide = (
    period_percent
    .T
    .reset_index()
)

period_wide = period_wide.rename(
    columns={
        "Topic": "Topic"
    }
)

period_wide["SubstantiveTopic"] = (
    period_wide["Topic"]
    .map(SUBSTANTIVE_LABELS)
)

period_wide["AbsoluteChange"] = (
    period_wide[
        "Post-Acute COVID Period (2022-2025)"
    ]
    -
    period_wide[
        "Pre-COVID (2016-2019)"
    ]
)

period_wide["COVIDPeriodIncrease"] = (
    period_wide[
        "COVID-19 Acute Period (2020-2021)"
    ]
    -
    period_wide[
        "Pre-COVID (2016-2019)"
    ]
)

period_wide = period_wide.sort_values(
    "AbsoluteChange",
    ascending=False
)

period_wide.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "04_Topic_Change_PreCOVID_to_PostCOVID.xlsx"
    ),
    index=False
)


# ============================================================
# 17. COVID-19 DOCUMENT IDENTIFICATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 7. COVID-19 analysis")
print("=" * 70)


COVID_PATTERN = re.compile(
    r"""
    \bcovid(?:-|\s)?19\b
    |
    \bcovid\b
    |
    \bsars[-\s]?cov[-\s]?2\b
    |
    \b2019[-\s]?ncov\b
    |
    \bcoronavirus(?:\s+disease)?\b
    |
    \bsevere acute respiratory syndrome coronavirus 2\b
    """,
    flags=re.IGNORECASE |
          re.VERBOSE
)


analysis_df["COVID19"] = (
    analysis_df["CombinedText"]
    .fillna("")
    .str.contains(
        COVID_PATTERN,
        regex=True,
        na=False
    )
)


print(
    "COVID-19 documents:",
    analysis_df["COVID19"].sum()
)

print(
    "COVID-19 percentage:",
    round(
        analysis_df["COVID19"].mean() * 100,
        2
    ),
    "%"
)


# ============================================================
# 18. COVID BY YEAR
# ============================================================

covid_year = (
    analysis_df
    .groupby("Year")["COVID19"]
    .agg(
        Documents="sum",
        Total="count"
    )
    .reset_index()
)

covid_year["Percentage"] = (
    covid_year["Documents"]
    /
    covid_year["Total"]
    * 100
)

covid_year.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "05_COVID19_By_Year.xlsx"
    ),
    index=False
)


# ============================================================
# 19. COVID BY TOPIC
# ============================================================

covid_topic = (
    analysis_df
    .groupby("Topic")
    .agg(
        TotalDocuments=("Topic", "size"),
        COVID19Documents=("COVID19", "sum")
    )
    .reset_index()
)

covid_topic["COVID19Percentage"] = (
    covid_topic["COVID19Documents"]
    /
    covid_topic["TotalDocuments"]
    * 100
)

covid_topic["SubstantiveTopic"] = (
    covid_topic["Topic"]
    .map(SUBSTANTIVE_LABELS)
)

covid_topic = covid_topic.sort_values(
    "COVID19Documents",
    ascending=False
)

covid_topic.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "06_COVID19_By_Topic.xlsx"
    ),
    index=False
)


# ============================================================
# 20. COVID TOPIC DISTRIBUTION
# ============================================================

covid_topic_distribution = (
    analysis_df[
        analysis_df["COVID19"]
    ]
    .groupby("Topic")
    .size()
    .reset_index(
        name="COVID19Documents"
    )
)

covid_topic_distribution["SubstantiveTopic"] = (
    covid_topic_distribution["Topic"]
    .map(SUBSTANTIVE_LABELS)
)

covid_topic_distribution[
    "PercentageOfCOVIDDocuments"
] = (
    covid_topic_distribution["COVID19Documents"]
    /
    covid_topic_distribution["COVID19Documents"].sum()
    * 100
)

covid_topic_distribution = (
    covid_topic_distribution
    .sort_values(
        "COVID19Documents",
        ascending=False
    )
)

covid_topic_distribution.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "07_COVID19_Topic_Distribution.xlsx"
    ),
    index=False
)


# ============================================================
# 21. AI-RELATED DOCUMENT IDENTIFICATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 8. AI-related research analysis")
print("=" * 70)


AI_PATTERN = re.compile(
    r"""
    \bartificial intelligence\b
    |
    \bmachine learning\b
    |
    \bdeep learning\b
    |
    \bneural network(?:s)?\b
    |
    \bcomputer vision\b
    |
    \bnatural language processing\b
    |
    \blarge language model(?:s)?\b
    |
    \bchatgpt\b
    |
    \bgenerative ai\b
    |
    \bgenerative artificial intelligence\b
    |
    \balgorithm(?:ic)?\b
    |
    \bpredictive model(?:ing)?\b
    |
    \bautomated detection\b
    |
    \bintelligent system(?:s)?\b
    |
    \bsmart monitoring\b
    """,
    flags=re.IGNORECASE |
          re.VERBOSE
)


analysis_df["AI_Related"] = (
    analysis_df["CombinedText"]
    .fillna("")
    .str.contains(
        AI_PATTERN,
        regex=True,
        na=False
    )
)


print(
    "AI-related documents:",
    analysis_df["AI_Related"].sum()
)

print(
    "AI-related percentage:",
    round(
        analysis_df["AI_Related"].mean() * 100,
        2
    ),
    "%"
)


# ============================================================
# 22. AI BY YEAR
# ============================================================

ai_year = (
    analysis_df
    .groupby("Year")["AI_Related"]
    .agg(
        Documents="sum",
        Total="count"
    )
    .reset_index()
)

ai_year["Percentage"] = (
    ai_year["Documents"]
    /
    ai_year["Total"]
    * 100
)

ai_year.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "08_AI_Related_By_Year.xlsx"
    ),
    index=False
)


# ============================================================
# 23. AI BY TOPIC
# ============================================================

ai_topic = (
    analysis_df
    .groupby("Topic")
    .agg(
        TotalDocuments=("Topic", "size"),
        AIRelatedDocuments=("AI_Related", "sum")
    )
    .reset_index()
)

ai_topic["AIRelatedPercentage"] = (
    ai_topic["AIRelatedDocuments"]
    /
    ai_topic["TotalDocuments"]
    * 100
)

ai_topic["SubstantiveTopic"] = (
    ai_topic["Topic"]
    .map(SUBSTANTIVE_LABELS)
)

ai_topic = ai_topic.sort_values(
    "AIRelatedDocuments",
    ascending=False
)

ai_topic.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "09_AI_Related_By_Topic.xlsx"
    ),
    index=False
)


# ============================================================
# 24. AI TOPIC DISTRIBUTION
# ============================================================

ai_topic_distribution = (
    analysis_df[
        analysis_df["AI_Related"]
    ]
    .groupby("Topic")
    .size()
    .reset_index(
        name="AIRelatedDocuments"
    )
)

if len(ai_topic_distribution) > 0:

    ai_topic_distribution[
        "SubstantiveTopic"
    ] = (
        ai_topic_distribution["Topic"]
        .map(SUBSTANTIVE_LABELS)
    )

    ai_topic_distribution[
        "PercentageOfAIDocuments"
    ] = (
        ai_topic_distribution[
            "AIRelatedDocuments"
        ]
        /
        ai_topic_distribution[
            "AIRelatedDocuments"
        ].sum()
        * 100
    )

    ai_topic_distribution = (
        ai_topic_distribution
        .sort_values(
            "AIRelatedDocuments",
            ascending=False
        )
    )

else:

    ai_topic_distribution = pd.DataFrame(
        columns=[
            "Topic",
            "AIRelatedDocuments",
            "SubstantiveTopic",
            "PercentageOfAIDocuments"
        ]
    )


ai_topic_distribution.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "10_AI_Related_Topic_Distribution.xlsx"
    ),
    index=False
)


# ============================================================
# 25. COVID × AI × TOPIC MATRIX
# ============================================================

event_topic = (
    analysis_df
    .groupby("Topic")
    .agg(
        Documents=("Topic", "size"),
        COVID19=("COVID19", "sum"),
        AI_Related=("AI_Related", "sum")
    )
    .reset_index()
)

event_topic["SubstantiveTopic"] = (
    event_topic["Topic"]
    .map(SUBSTANTIVE_LABELS)
)

event_topic["COVID19_Percent"] = (
    event_topic["COVID19"]
    /
    event_topic["Documents"]
    * 100
)

event_topic["AI_Percent"] = (
    event_topic["AI_Related"]
    /
    event_topic["Documents"]
    * 100
)

event_topic.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "11_COVID_AI_Topic_Matrix.xlsx"
    ),
    index=False
)


# ============================================================
# 26. JOURNAL × TOPIC ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("STEP 9. Journal-specific thematic analysis")
print("=" * 70)


journal_topic_counts = pd.crosstab(
    analysis_df["Journal"],
    analysis_df["Topic"]
)

journal_topic_counts = journal_topic_counts.reindex(
    index=JOURNALS,
    fill_value=0
)

journal_topic_counts = journal_topic_counts.reindex(
    columns=all_topics,
    fill_value=0
)

journal_topic_percent = (
    journal_topic_counts
    .div(
        journal_topic_counts.sum(axis=1),
        axis=0
    )
    * 100
)

journal_topic_counts.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "12_Journal_Topic_Counts.xlsx"
    )
)

journal_topic_percent.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "13_Journal_Topic_Percent.xlsx"
    )
)


# ============================================================
# 27. LONG JOURNAL × TOPIC TABLE
# ============================================================

journal_topic_long = (
    journal_topic_percent
    .reset_index()
    .melt(
        id_vars="Journal",
        var_name="Topic",
        value_name="Percentage"
    )
)

journal_topic_long["Topic"] = (
    journal_topic_long["Topic"]
    .astype(int)
)

journal_topic_long["SubstantiveTopic"] = (
    journal_topic_long["Topic"]
    .map(SUBSTANTIVE_LABELS)
)

journal_topic_long.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "14_Journal_Topic_Long.xlsx"
    ),
    index=False
)


# ============================================================
# 28. YEAR × JOURNAL × TOPIC
# ============================================================

print("\nCalculating Year × Journal × Topic matrix...")


year_journal_topic = (
    analysis_df
    .groupby(
        [
            "Year",
            "Journal",
            "Topic"
        ]
    )
    .size()
    .reset_index(
        name="Documents"
    )
)

year_journal_topic["SubstantiveTopic"] = (
    year_journal_topic["Topic"]
    .map(SUBSTANTIVE_LABELS)
)

year_journal_topic.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "15_Year_Journal_Topic.xlsx"
    ),
    index=False
)


# ============================================================
# 29. TOPICS OVER TIME
# ============================================================

print("\n" + "=" * 70)
print("STEP 10. BERTopic Topics Over Time")
print("=" * 70)

docs = analysis_df["CombinedText"].tolist()

timestamps = analysis_df["Year"].tolist()

print(
    "Documents:",
    len(docs)
)

print(
    "Timestamps:",
    len(timestamps)
)


topics_over_time = None

try:

    topics_over_time = topic_model.topics_over_time(
        docs,
        timestamps,
        nr_bins=None,
        evolution_tuning=True,
        global_tuning=True
    )

    print(
        "\nTopics-over-time calculation completed."
    )

    topics_over_time.to_excel(
        os.path.join(
            OUTPUT_DIR,
            "16_Topics_Over_Time.xlsx"
        ),
        index=False
    )

except Exception as e:

    print(
        "\nWARNING: topics_over_time failed."
    )

    print(
        "Reason:",
        str(e)
    )

    print(
        "Annual prevalence analysis remains valid."
    )


# ============================================================
# 30. VISUALIZE TOPICS OVER TIME
# ============================================================

if topics_over_time is not None:

    try:

        fig = topic_model.visualize_topics_over_time(
            topics_over_time,
            top_n_topics=20
        )

        fig.write_html(
            os.path.join(
                OUTPUT_DIR,
                "17_Topics_Over_Time_Top20.html"
            )
        )

        print(
            "Saved: 17_Topics_Over_Time_Top20.html"
        )

    except Exception as e:

        print(
            "Topics-over-time visualization failed:"
        )

        print(
            str(e)
        )


# ============================================================
# 31. MAJOR TOPIC TRAJECTORIES
# ============================================================

print("\n" + "=" * 70)
print("STEP 11. Creating major topic trajectory figure")
print("=" * 70)


top_topics = (
    topic_summary[
        topic_summary["Topic"] >= 0
    ]
    .sort_values(
        "Documents",
        ascending=False
    )
    .head(15)["Topic"]
    .tolist()
)


plt.figure(
    figsize=(14, 9)
)

for topic in top_topics:

    subset = annual_percent[
        topic
    ]

    label = SUBSTANTIVE_LABELS.get(
        topic,
        f"Topic {topic}"
    )

    plt.plot(
        YEARS,
        subset.values,
        marker="o",
        linewidth=1.5,
        label=f"T{topic}: {label}"
    )

plt.xlabel(
    "Publication year"
)

plt.ylabel(
    "Topic prevalence (%)"
)

plt.title(
    "Temporal Evolution of Major Occupational Health Research Topics, 2016-2025"
)

plt.xticks(
    YEARS,
    rotation=45
)

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=8
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "Figure_1_Major_Topic_Trajectories.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 32. COVID TRAJECTORY FIGURE
# ============================================================

print(
    "Creating COVID-19 trajectory figure..."
)


covid_topic_year = (
    analysis_df[
        analysis_df["Topic"] == 0
    ]
    .groupby("Year")
    .size()
    .reindex(
        YEARS,
        fill_value=0
    )
)

total_year = (
    analysis_df
    .groupby("Year")
    .size()
    .reindex(
        YEARS,
        fill_value=0
    )
)

covid_topic_year_percent = (
    covid_topic_year
    /
    total_year
    * 100
)


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    YEARS,
    covid_topic_year_percent.values,
    marker="o",
    linewidth=2
)

plt.axvline(
    2020,
    linestyle="--",
    linewidth=1
)

plt.xlabel(
    "Publication year"
)

plt.ylabel(
    "Prevalence of COVID-19 topic (%)"
)

plt.title(
    "Emergence and Persistence of the COVID-19 Topic"
)

plt.xticks(
    YEARS,
    rotation=45
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "Figure_2_COVID19_Topic_Trajectory.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 33. COVID KEYWORD TRAJECTORY
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    YEARS,
    covid_year["Percentage"].values,
    marker="o",
    linewidth=2
)

plt.axvline(
    2020,
    linestyle="--",
    linewidth=1
)

plt.xlabel(
    "Publication year"
)

plt.ylabel(
    "Documents containing COVID-19 terminology (%)"
)

plt.title(
    "COVID-19-Related Occupational Health Research, 2016-2025"
)

plt.xticks(
    YEARS,
    rotation=45
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "Figure_3_COVID19_Keyword_Trajectory.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 34. AI TRAJECTORY
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    YEARS,
    ai_year["Percentage"].values,
    marker="o",
    linewidth=2
)

plt.axvline(
    2020,
    linestyle="--",
    linewidth=1
)

plt.xlabel(
    "Publication year"
)

plt.ylabel(
    "AI-related documents (%)"
)

plt.title(
    "Emergence of AI-Related Occupational Health Research, 2016-2025"
)

plt.xticks(
    YEARS,
    rotation=45
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "Figure_4_AI_Emergence.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 35. JOURNAL × TOPIC HEATMAP
# ============================================================

print(
    "Creating journal-topic heatmap..."
)


# Top 25 topics by corpus size
heatmap_topics = (
    topic_summary[
        topic_summary["Topic"] >= 0
    ]
    .sort_values(
        "Documents",
        ascending=False
    )
    .head(25)["Topic"]
    .tolist()
)

heatmap_data = (
    journal_topic_percent[
        heatmap_topics
    ]
)

heatmap_labels = [
    f"T{t}: {SUBSTANTIVE_LABELS.get(t, f'Topic {t}')}"
    for t in heatmap_topics
]


plt.figure(
    figsize=(14, 10)
)

plt.imshow(
    heatmap_data.values,
    aspect="auto"
)

plt.colorbar(
    label="Within-journal prevalence (%)"
)

plt.xticks(
    range(len(heatmap_topics)),
    [
        f"T{t}"
        for t in heatmap_topics
    ],
    rotation=90
)

plt.yticks(
    range(len(heatmap_data.index)),
    heatmap_data.index
)

plt.xlabel(
    "Topic"
)

plt.ylabel(
    "Journal"
)

plt.title(
    "Journal-Specific Thematic Profiles"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "Figure_5_Journal_Topic_Heatmap.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 36. REPRESENTATIVE DOCUMENTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 12. Extracting representative documents")
print("=" * 70)


representative_rows = []

for topic in range(53):

    topic_docs = analysis_df[
        analysis_df["Topic"] == topic
    ].copy()

    if len(topic_docs) == 0:
        continue

    # We use the shortest available text as a
    # practical representative-document pool.
    # The BERTopic model's own representative
    # documents remain available in the baseline output.

    topic_docs["TextLength"] = (
        topic_docs["CombinedText"]
        .str.len()
    )

    topic_docs = topic_docs.sort_values(
        "TextLength"
    ).head(10)

    for _, row in topic_docs.iterrows():

        title = ""

        if title_col is not None:
            title = str(
                row.get(
                    title_col,
                    ""
                )
            )

        pmid = ""

        if pmid_col is not None:
            pmid = str(
                row.get(
                    pmid_col,
                    ""
                )
            )

        representative_rows.append(
            {
                "Topic": topic,
                "SubstantiveTopic":
                    SUBSTANTIVE_LABELS.get(
                        topic,
                        f"Topic {topic}"
                    ),
                "PMID": pmid,
                "Year": row["Year"],
                "Journal": row["Journal"],
                "Title": title
            }
        )


representative_df = pd.DataFrame(
    representative_rows
)

representative_df.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "18_Representative_Document_Pool.xlsx"
    ),
    index=False
)


# ============================================================
# 37. FULL DOCUMENT-LEVEL DATASET
# ============================================================

analysis_output_columns = [
    c for c in [
        pmid_col,
        doi_col,
        title_col,
        "Journal",
        "Year",
        "Topic",
        "SubstantiveTopic",
        "HistoricalPeriod",
        "COVID19",
        "AI_Related"
    ]
    if c is not None and c in analysis_df.columns
]

analysis_df[
    analysis_output_columns
].to_excel(
    os.path.join(
        OUTPUT_DIR,
        "19_Document_Level_Analysis_Data.xlsx"
    ),
    index=False
)


# ============================================================
# 38. MASTER EXCEL WORKBOOK
# ============================================================

print("\n" + "=" * 70)
print("STEP 13. Creating master Excel workbook")
print("=" * 70)


MASTER_FILE = os.path.join(
    OUTPUT_DIR,
    "Occupational_Health_BERTopic_Temporal_Analysis.xlsx"
)


with pd.ExcelWriter(
    MASTER_FILE,
    engine="openpyxl"
) as writer:

    topic_summary.to_excel(
        writer,
        sheet_name="Topic_Summary",
        index=False
    )

    annual_long.to_excel(
        writer,
        sheet_name="Annual_Prevalence",
        index=False
    )

    period_long.to_excel(
        writer,
        sheet_name="Period_Prevalence",
        index=False
    )

    period_wide.to_excel(
        writer,
        sheet_name="Topic_Change",
        index=False
    )

    covid_year.to_excel(
        writer,
        sheet_name="COVID_By_Year",
        index=False
    )

    covid_topic.to_excel(
        writer,
        sheet_name="COVID_By_Topic",
        index=False
    )

    covid_topic_distribution.to_excel(
        writer,
        sheet_name="COVID_Topic_Distribution",
        index=False
    )

    ai_year.to_excel(
        writer,
        sheet_name="AI_By_Year",
        index=False
    )

    ai_topic.to_excel(
        writer,
        sheet_name="AI_By_Topic",
        index=False
    )

    ai_topic_distribution.to_excel(
        writer,
        sheet_name="AI_Topic_Distribution",
        index=False
    )

    event_topic.to_excel(
        writer,
        sheet_name="COVID_AI_Matrix",
        index=False
    )

    journal_topic_long.to_excel(
        writer,
        sheet_name="Journal_Topic",
        index=False
    )

    year_journal_topic.to_excel(
        writer,
        sheet_name="Year_Journal_Topic",
        index=False
    )

    representative_df.to_excel(
        writer,
        sheet_name="Representative_Docs",
        index=False
    )

    analysis_df[
        analysis_output_columns
    ].to_excel(
        writer,
        sheet_name="Document_Level",
        index=False
    )

    if topics_over_time is not None:

        topics_over_time.to_excel(
            writer,
            sheet_name="Topics_Over_Time",
            index=False
        )


# ============================================================
# 39. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\nFinal corpus:",
    len(analysis_df)
)

print(
    "Substantive topics:",
    53
)

print(
    "Outlier documents:",
    (analysis_df["Topic"] == -1).sum()
)

print(
    "Outlier percentage:",
    round(
        (
            (analysis_df["Topic"] == -1).sum()
            /
            len(analysis_df)
            * 100
        ),
        2
    ),
    "%"
)

print(
    "\nCOVID-19 documents:",
    analysis_df["COVID19"].sum()
)

print(
    "COVID-19 percentage:",
    round(
        analysis_df["COVID19"].mean() * 100,
        2
    ),
    "%"
)

print(
    "\nAI-related documents:",
    analysis_df["AI_Related"].sum()
)

print(
    "AI-related percentage:",
    round(
        analysis_df["AI_Related"].mean() * 100,
        2
    ),
    "%"
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print(
    "\nMaster Excel:"
)

print(
    MASTER_FILE
)

print(
    "\nKey figures:"
)

for filename in [
    "Figure_1_Major_Topic_Trajectories.png",
    "Figure_2_COVID19_Topic_Trajectory.png",
    "Figure_3_COVID19_Keyword_Trajectory.png",
    "Figure_4_AI_Emergence.png",
    "Figure_5_Journal_Topic_Heatmap.png"
]:

    print(
        os.path.join(
            OUTPUT_DIR,
            filename
        )
    )

print(
    "\n" + "=" * 70
)
print("END")
print("=" * 70)