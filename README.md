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

The core hypothesis of the imitation model is that a new game's description being semantically similar to descriptions of *previously successful* games is a meaningful signal of its own potential success. To test this, each game's description is encoded into a 384-dimensional vector using the `all-MiniLM-L6-v2` SBERT model (encoded in batches of 32 across the full ~114k game corpus).

**Description preprocessing** is applied before encoding to reduce noise from re-releases of already-successful games. The following edition phrases are stripped and text is lowercased and punctuation-normalized:
- "Collector's Edition", "Complete Edition", "Game of the Year Edition", "Definitive Edition"

Without this step, a remaster of a well-known hit would score artificially high similarity to the original, inflating the feature for a game that isn't independently successful.

**Chronological integrity** is preserved throughout: the pipeline iterates through games in release-date order, maintaining a growing list of past embeddings. When computing similarity for any given game, it only compares against games released *before* it — no future information leaks into the features.

**Feature construction** for each game:

- **`sim_to_past_hits`** — Among all past hit games, finds the top-5 most similar by cosine similarity, then computes a time-weighted average of those similarity scores. Weights decay with the age of the comparable hit: `weight = 1 / (1 + days_since_that_hit_released)`, so a game similar to a recent hit scores higher than one similar to a hit from a decade ago.
- **`log_days_since_similar_hit`** — Log-scaled number of days between the current game and its single closest hit neighbor (the argmax match, used separately from the weighted average above).
- **`sim_dev_interaction`** — `sim_to_past_hits × dev_success`: captures whether a game is both semantically similar to hits *and* made by a developer with a proven track record.
- **`sim_time_interaction`** — `sim_to_past_hits × (1 / (1 + log_days_since_similar_hit))`: down-weights similarity to hits that are very old relative to the current game.

These four features are added on top of the clean structured feature set and fed into the same Logistic Regression classifier.

---

## Models

All models are trained on a chronological 70/30 split with `StandardScaler` preprocessing. The primary classifier is **Logistic Regression** with `class_weight='balanced'` and a decision threshold tuned to 0.35 to improve recall on the minority hit class. Three alternative models were also evaluated: Logistic Regression + PCA, LASSO Regression, and a Support Vector Machine.

### Full Results (TABLE I from project report)

| Model | Feature Set | F1 Score | Average Precision |
|-------|-------------|----------|-------------------|
| Logistic Regression | Metadata Baseline | 0.212 | 0.12 |
| Logistic Regression | Similarity Features | 0.217 | 0.12 |
| Logistic Regression + PCA | Metadata + PCA Embeddings | 0.229 | 0.15 |
| LASSO Regression | Metadata Baseline | 0.215 | 0.12 |
| Support Vector Machine | Metadata Baseline | 0.185 | 0.11 |

### PCA Model

Rather than compressing SBERT embeddings into four handcrafted similarity features, the PCA model feeds the full 384-dimensional sentence embeddings through scikit-learn's `PCA`, then concatenates the reduced components directly to the structured metadata. The number of components was swept from 40 to 60; **46 components was optimal**, capturing enough semantic variance without overfitting. This approach achieves the highest F1 (0.229) and the highest Average Precision (0.15) of any model tested — suggesting that PCA-reduced embeddings preserve richer semantic relationships than the handcrafted features can represent.

### Alternative Models

**LASSO Regression** (F1 = 0.215) marginally outperforms the metadata baseline, indicating some benefit from regularized feature selection. The **Support Vector Machine** performs worst of all tested models (F1 = 0.185), which is a meaningful finding: if a kernel SVM cannot improve on linear models, it suggests there are no strong non-linear decision boundaries in the data — the feature relationships are largely linear.

### Feature Importance (Similarity Model)

The top coefficients for the Logistic Regression trained on similarity features:

| Feature | Coefficient |
|---------|-------------|
| `dev_success` | 0.458 |
| `sim_to_past_hits` | 0.196 |
| Free To Play | 0.187 |
| `dev_had_success` | 0.170 |
| RPG | 0.159 |
| Simulation | 0.108 |
| Strategy | 0.102 |

Developer history (`dev_success`) is by far the strongest predictor, with the SBERT similarity feature (`sim_to_past_hits`) as the second-most influential non-metadata signal.

### New Developer Test

To assess how the model performs in the absence of developer history — the most important feature — it was evaluated exclusively on games by developers with no prior releases in the dataset. F1 dropped sharply to **0.12**, confirming that the model's predictive power depends heavily on the `dev_success` signal and degrades significantly for debut developers.

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
