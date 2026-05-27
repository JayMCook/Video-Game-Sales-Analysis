# Steam Game Success Prediction Pipeline

A machine learning pipeline that predicts whether an upcoming Steam game will become a "hit" (≥50,000 owners) using pre-release metadata and semantic similarity to past successful games.

📄 [Full project report](ML%20Final%20Report.pdf)

---

## Overview

The Steam storefront hosts over 100,000 games, with the vast majority going unnoticed. This project builds a binary classifier to evaluate a game's likelihood of success *before* it releases, using only information available at launch — price, genre, developer history, and textual description.

The pipeline has two layers:

1. **Baseline model** — Logistic Regression on structured metadata (developer history, genre, price, year)
2. **Imitation model** — Adds SBERT-based semantic similarity features that measure how closely a new game's description resembles past hits

---

## Dataset

- **Source**: Steam game metadata (~114,000 games after cleaning)
- **Target variable**: Binary "Hit" label — `1` if estimated owners ≥ 50,000, `0` otherwise
- **Class distribution**: ~12.4% hits, ~87.6% misses (heavily imbalanced)
- **Train/test split**: Chronological 70/30 — data is sorted by release date and split in order, simulating prediction of future games rather than random held-out samples

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01-data-cleaning.ipynb` | Data cleaning, feature engineering, target construction |
| `02-baseline-model.ipynb` | Logistic regression baseline, data leakage discovery & fix, threshold tuning |
| `03-model-analysis.ipynb` | Feature ablation study (dev features vs. genre features vs. full feature set) |
| `04-imitation-modeling.ipynb` | SBERT embedding pipeline, cosine similarity to past hits, final model |

---

## Feature Engineering

### Structured Features
- **`price`** — Game price at release
- **`year`** — Release year
- **`is_free`** — Binary flag for free-to-play games
- **`genre_*`** — One-hot encoded genre columns (e.g., Action, RPG, Simulation)
- **`dev_success`** — Developer's historical hit rate on prior releases (computed chronologically to avoid leakage)
- **`dev_had_success`** — Binary flag: has this developer released a hit before?
- **`log_dev_experience`** — Log-scaled count of the developer's prior releases
- **`num_devs`** — Number of developers credited on the game

> **Note on data leakage**: An early version of the model included `num_tags`, which appeared predictive but is a user-submitted feature populated *after* release. Removing it for the clean model reduced F1 but produced an honest predictor.

### Semantic Similarity Features (Imitation Model)
Using the `all-MiniLM-L6-v2` SBERT model, each game's description is encoded into a 384-dimensional vector. For each game, the pipeline computes:

- **`sim_to_past_hits`** — Time-weighted cosine similarity to the top-5 most similar prior hit games. Recent hits are weighted more heavily: `weight = 1 / (1 + days_since_release)`
- **`log_days_since_similar_hit`** — Log-scaled time gap between the current game and its closest hit neighbor
- **`sim_dev_interaction`** — `sim_to_past_hits × dev_success`
- **`sim_time_interaction`** — `sim_to_past_hits × time_decay_weight`

Re-release edition phrases ("Game of the Year Edition", "Definitive Edition", etc.) are stripped from descriptions before encoding to avoid similarity inflation from remasters of already-successful games.

---

## Model

**Logistic Regression** with `class_weight='balanced'` and `StandardScaler` preprocessing.

- `class_weight='balanced'` compensates for the 88/12 class imbalance
- Decision threshold tuned to 0.35 (vs. default 0.5) to improve F1 on the minority class
- Train/test split is **time-based**, not random — this prevents future data from leaking into training

### Baseline Model Results (clean features, no leakage)

| Metric | Score |
|--------|-------|
| Accuracy | ~94% |
| F1 (hits) | ~0.21 |
| Precision (hits) | ~0.19 |
| Recall (hits) | ~0.23 |

### Feature Ablation Summary

| Feature Set | F1 (hits) | Notes |
|-------------|-----------|-------|
| Dev features only | 0.207 | Developer history alone |
| Genre features only | 0.087 | High recall, very low precision |
| All features (leaky) | 0.302 | Includes `num_tags` — post-release data |
| Clean features | 0.212 | Pre-release features only |

---

## Tech Stack

- Python 3.13
- pandas, numpy
- scikit-learn (`LogisticRegression`, `StandardScaler`, `MultiLabelBinarizer`)
- `sentence-transformers` (`all-MiniLM-L6-v2`)
- matplotlib

---

## Setup

```bash
# Clone the repo
git clone https://github.com/JayMCook/Video-Game-Sales-Analysis.git
cd Video-Game-Sales-Analysis

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn sentence-transformers matplotlib jupyter

# Add your Steam dataset
# Place games.json in a /data directory at the project root

# Run the notebooks in order
jupyter notebook
```

> The notebooks reference `../data/games.json` and write intermediate files (`cleaned_games.pkl`, `genre_columns.pkl`) to `../data/`. Make sure that directory exists before running.

---

## Project Structure

```
Video-Game-Sales-Analysis/
├── notebooks/
│   ├── 01-data-cleaning.ipynb
│   ├── 02-baseline-model.ipynb
│   ├── 03-model-analysis.ipynb
│   └── 04-imitation-modeling.ipynb
├── data/               # Not tracked in git (add games.json here)
├── ML Final Report.pdf
└── README.md
```
