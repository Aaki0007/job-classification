# OSINT Labor Market Intelligence — Job Experience Level Classification

Classifying LinkedIn job postings into experience levels (junior, mid, senior) using semantic community detection and zero-shot LLM classification. The project discovers natural job market clusters via sentence embeddings and Louvain community detection, then uses Mistral Nemo Instruct to classify each community's experience level and entry-point accessibility.

## Dataset

LinkedIn Job Postings dataset from Kaggle (`arshkon/linkedin-job-postings`), containing ~124k postings. After cleaning and filtering (removing rows missing title/description/experience level, filtering descriptions to 400–700 words), we work with ~36k postings.

Columns used:
- `company_name` — name of the hiring company
- `title` — job title
- `description` — full job description text
- `location` — job location
- `formatted_work_type` — employment type (full-time, part-time, contract, etc.)
- `formatted_experience_level` — original experience level label from LinkedIn
- `skills_desc` — required skills
- `normalized_salary` — normalized salary figure

Dropped columns: `currency` and `compensation_type` were removed due to lack of diversity (only USD and base salary).

Experience level mapping:
- **junior**: Entry level, Internship, Associate
- **mid**: Mid-Senior level
- **senior**: Director, Executive, Principal, Lead, Manager

Final distribution: ~19.5k junior, ~15.1k mid, ~1.6k senior.

## Pipeline

```
Raw Data (LinkedIn postings)
        ↓
Notebook 1: Clean, engineer features → ~36k cleaned postings
        ↓
Notebook 2: Generate embeddings → Louvain community detection → 50 communities
        ↓
Notebook 3: Parallel LLM classification (50 communities × 3 attempts)
        ↓
Notebook 4: Merge results, aggregate votes, rank by entry accessibility
        ↓
Final Intelligence: Labeled dataset with entry barriers, skill requirements,
                    confidence scores, and actionable market insights
```

### Stage 1 — EDA & Feature Engineering

**Notebook:** `notebooks/Notebook_1_EDA_Feature_Engineering_v3/Notebook_1_EDA_Feature_Engineering.ipynb`

- Loads raw data, removes duplicates, drops rows missing title/description/experience level
- Filters by description length (400–700 words)
- Extracts engineered features:
  - **Years of experience** — regex-extracted from description text, binned into junior (0–2), mid (3–5), senior (6+)
  - **Title-based seniority** — keyword matching on job title (e.g., "intern", "lead", "director")
  - **Salary signal** — binned salary ranges (<50k junior, 50–100k mid, >100k senior)
  - **Skill complexity** — count and categorization of skills (advanced vs mid-level keywords)
- Outputs `postings_for_semantic_pipeline.csv` for downstream stages

### Stage 2 — Semantic Community Detection

**Notebook:** `notebooks/Notebook_2_Perfect_OSINT_Community_Detection/Notebook_2_Perfect_OSINT_Community_Detection.ipynb`

- Generates semantic embeddings using **all-MiniLM-L6-v2** sentence transformer
- Builds cosine similarity graph between job posting embeddings
- Runs **Louvain community detection** to identify 50 natural semantic clusters
- Tunes community detection hyperparameters across a resolution grid
- Outputs community summaries, cards, and tuning results

### Stage 3 — LLM-Based Community Classification

**Notebook:** `notebooks/Notebook_3_Grammar_First_Stage3_Intelligence/` (parallelized across batch subdirectories)

- Uses **Mistral Nemo Instruct** (Q5_K_M quantization, 8.13 GB) via llama.cpp with CUDA
- Classifies each of 50 communities along two dimensions:
  1. **Experience level distribution** — dominant level (junior / mid / senior) based on language patterns and experience share
  2. **Entry-point accessibility**:
     - *clear_entry_point*: >60% junior roles, no credentials required
     - *moderate_entry_barrier*: mixed experience or junior with credential preferences
     - *restricted_entry_point*: senior roles dominate or credentials mandatory
- Extracts community labels, defining skills, responsibility signals, and confidence levels
- Runs 3 classification attempts per community for ensemble voting
- Parallelized across multiple Colab GPU instances (one per batch range)

### Stage 4 — Merging & Actionable Intelligence

**Notebook:** `notebooks/Notebook_4_merging/Merging_and_stage_3_intelligence.ipynb`

- Merges all 50 batch results into a unified dataset
- Aggregates multiple LLM classification attempts via voting
- Computes agreement metrics: `level_agreement`, `access_agreement`, `label_agreement`, `overall_agreement`
- Produces ranked/filtered datasets:
  - `entry_point_communities_ranked.csv` — communities ranked by entry accessibility
  - `restricted_entry_communities_ranked.csv` — high-barrier entry communities
  - `ambiguous_communities_for_review.csv` — communities with conflicting signals needing human review

## Project Structure

```
notebooks/
  Notebook_1_EDA_Feature_Engineering_v3/
    Notebook_1_EDA_Feature_Engineering.ipynb   # Stage 1: EDA, cleaning, feature engineering
    input data/postings.csv                    # Raw LinkedIn data
    results/                                   # Cleaned output CSVs
    requirements.txt                           # Stage 1 dependencies
  Notebook_2_Perfect_OSINT_Community_Detection/
    Notebook_2_Perfect_OSINT_Community_Detection.ipynb  # Stage 2: Embeddings & community detection
    community_summary.csv                      # Community-level statistics
    community_cards.json                       # Community metadata
    requirements.txt                           # Stage 2 dependencies
  Notebook_3_Grammar_First_Stage3_Intelligence/
    {1,2-6,7-11,...,51-53}/                    # Batch subdirectories for parallel LLM inference
      prompt_python.py                         # LLM prompt template
      results-*/                               # Per-batch classification results
  Notebook_4_merging/
    Merging_and_stage_3_intelligence.ipynb      # Stage 4: Merge, aggregate, rank
    results/                                   # Final intelligence outputs
    requirements.txt                           # Stage 4 dependencies
```

## Reproducibility

Each notebook directory includes reproducibility metadata:
- `requirements.txt` — pinned Python package versions
- `environment_log*.json` — full environment specifications
- `reproducibility_manifest.json` — checksums and pipeline metadata
- `seed_log*.json` — random seed values used
- `dataset_hash.json` / `input_hashes*.json` — data integrity hashes

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (designed for Google Colab GPU runtime)
- llama-cpp-python (built with CUDA)
- pandas, numpy, scikit-learn, scipy
- torch, sentence-transformers (all-MiniLM-L6-v2)
- tqdm, matplotlib, psutil
- kagglehub, huggingface_hub

Each notebook directory contains its own `requirements.txt` with pinned versions for that stage.

## Usage

1. Run **Notebook 1** to clean the raw data and generate engineered features
2. Run **Notebook 2** to generate semantic embeddings and detect communities
3. Run **Notebook 3** batch notebooks on separate Colab GPU instances (one per batch range) to classify communities
4. Run **Notebook 4** to merge all results, compute agreement scores, and produce ranked intelligence outputs

## License

GPL-3.0 — see [LICENSE](LICENSE) for details.
