# Job Experience Level Classification

Classifying LinkedIn job postings into experience levels (junior, mid, senior) using zero-shot classification with a local LLM. The project uses Mistral Nemo Instruct as the classifier and evaluates its predictions against ground truth labels from the dataset.

## Dataset

LinkedIn Job Postings dataset from Kaggle (`arshkon/linkedin-job-postings`), containing ~124k postings. After cleaning and filtering (removing rows missing title/description/experience level, filtering descriptions to 400-700 words), we work with ~36k postings.

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

## Feature Engineering

The following engineered features are extracted and used as supporting signals during classification:

- **Years of experience** — regex-extracted from description text, binned into junior (0-2), mid (3-5), senior (6+)
- **Title-based seniority** — keyword matching on job title (e.g., "intern", "lead", "director")
- **Salary signal** — binned salary ranges (<50k junior, 50-100k mid, >100k senior)
- **Skill complexity** — count and categorization of skills (advanced vs mid-level keywords)

## Pipeline

1. **Data Cleaning** (`notebooks/Eda_and_feature_enginnering.ipynb`) — Load raw data, remove duplicates, drop rows missing title/description/experience level, filter by description length (400-700 words)
2. **Feature Engineering** (`notebooks/Eda_and_feature_enginnering.ipynb`) — Extract years of experience, title signals, salary signals, skill complexity indicators
3. **Label Standardization** — Map LinkedIn's experience levels into three categories: junior, mid, senior
4. **Data Chunking** — Split the ~36k dataset into 13 stratified chunks for parallel processing across multiple Colab instances
5. **Model Setup** — Download and load Mistral Nemo Instruct (Q5_K_M quantization, 8.13 GB) via llama.cpp with CUDA support
6. **Zero-Shot Classification** (`notebooks/chunking-*.ipynb`) — Classify each posting using a structured prompt that includes the job text and engineered feature hints, with JSON output parsing
7. **Result Combination** (`scripts/combine the chuncks.py`) — Merge all 13 chunk results into a single dataset
8. **Evaluation** — Compare predictions to ground truth using accuracy, classification report, confusion matrix, and a one-sided z-test (H0: accuracy <= 0.80)
9. **Semantic Clustering** — Generate embeddings with all-MiniLM-L6-v2, build cosine similarity graph, run Louvain community detection
10. **Actionable Intelligence** — Identify ambiguous clusters, misclassification patterns, and hidden structure within experience levels

## Project Structure

```
notebooks/
  Job_Experience.ipynb              # Initial prototyping (sample of 100 jobs)
  Eda_and_feature_enginnering.ipynb # EDA, cleaning, and feature engineering
  chunking-{2,3,5,6,7,9,12,13}.ipynb # Parallel inference workers (one per Colab instance)
scripts/
  combine the chuncks.py            # Merge chunk results and run evaluation + z-test
data/                                # Result CSVs (gitignored)
```

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (designed for Google Colab GPU runtime)
- llama-cpp-python (built with CUDA)
- pandas, numpy, scikit-learn, statsmodels
- kagglehub, huggingface_hub

Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run `notebooks/Eda_and_feature_enginnering.ipynb` to clean the data and generate engineered features
2. Run each `notebooks/chunking-*.ipynb` notebook on a separate Colab GPU instance (change `CHUNK_ID` per notebook)
3. Collect all `results_chunk_*.csv` files into `data/`
4. Run `scripts/combine the chuncks.py` to merge results and evaluate

## License

GPL-3.0 — see [LICENSE](LICENSE) for details.
