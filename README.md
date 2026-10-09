# News Popularity in Multiple Social Media Platforms

A machine learning regression project that predicts the **sentiment score of a news title** using text features (TF-IDF) and social media share counts (Facebook, Google+, LinkedIn).

**Author:** ABINAYA S
**Class:** III BCA A

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Models and Evaluation Metrics](#models-and-evaluation-metrics)
- [Results](#results)
- [Best Model](#best-model)
- [Conclusion](#conclusion)
- [Future Scope](#future-scope)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)

---

## Overview

News articles are shared widely across social platforms, and the tone of a headline can influence how readers respond. This project cleans and explores a large news dataset, converts the text into numerical features, and compares four regression algorithms to predict `SentimentTitle`.

## Dataset

- **Source:** [UCI Machine Learning Repository: News Popularity in Multiple Social Media Platforms](https://archive.ics.uci.edu/)
- **File used:** `news_popularity.csv`
- **Shape:** 93,239 rows × 11 columns
- **Topics include:** economy, Obama, Microsoft, Palestine

| Column | Type | Description |
|---|---|---|
| `IDLink` | float64 | Unique article ID |
| `Title` | object | News article title |
| `Headline` | object | Article text snippet |
| `Source` | object | Publisher of the article |
| `Topic` | object | Subject of the article |
| `PublishDate` | object | Date and time published |
| `SentimentTitle` | float64 | Sentiment score of the title (**target**) |
| `SentimentHeadline` | float64 | Sentiment score of the headline |
| `Facebook` | int64 | Share count on Facebook |
| `GooglePlus` | int64 | Share count on Google+ |
| `LinkedIn` | int64 | Share count on LinkedIn |

## Project Workflow

1. Load the dataset and inspect its shape, types and statistics
2. Clean the data (missing values and duplicates)
3. Explore the target variable and correlations
4. Apply a log transform to the share counts
5. Split the data into training and test sets
6. Vectorise text with TF-IDF
7. Train four regression models
8. Evaluate with MAE, MSE, RMSE and R²
9. Compare results and select the best model

### Data Cleaning

| Issue | Count | Action |
|---|---|---|
| Missing `Headline` | 15 | Filled with empty text |
| Missing `Source` | 279 | Filled with `"Unknown"` |
| Duplicate rows | 4 | Removed |

Shape after cleaning: **93,235 × 11**

## Exploratory Data Analysis

Summary of the target variable `SentimentTitle`:

| Statistic | Value |
|---|---|
| Count | 93,235 |
| Mean | -0.0054 |
| Std. deviation | 0.1364 |
| Median | 0.0000 |
| Minimum | -0.9507 |
| Maximum | 0.9624 |

Correlation with `SentimentTitle`:

| Feature | Correlation |
|---|---|
| `SentimentHeadline` | 0.1836 |
| `IDLink` | 0.0050 |
| `LinkedIn` | 0.0034 |
| `Facebook` | -0.0038 |
| `GooglePlus` | -0.0103 |

`SentimentHeadline` is the only noticeable linear relationship. Share counts show almost no direct linear correlation with title sentiment.

## Feature Engineering

- **Inputs (X):** `Title`, `Headline`, `SentimentHeadline`, `Facebook_log`, `GooglePlus_log`, `LinkedIn_log`
- **Target (y):** `SentimentTitle`
- **Log transform:** `log1p` applied to Facebook, GooglePlus and LinkedIn counts to reduce skew
- **Train–test split:** 80% / 20%, `random_state=42`

| Set | Shape |
|---|---|
| `X_train` | (74,588, 6) |
| `X_test` | (18,647, 6) |
| `y_train` | (74,588,) |
| `y_test` | (18,647,) |

**TF-IDF** (fitted on the training set only, then applied to the test set):

| Text field | Max features | Settings |
|---|---|---|
| Title | 30,000 | n-grams (1,2), `min_df=2`, `sublinear_tf=True` |
| Headline | 20,000 | n-grams (1,2), `min_df=2`, `sublinear_tf=True` |

The TF-IDF matrices are combined with the 4 numeric features using `hstack`:

- Final train matrix: **74,588 × 50,004**
- Final test matrix: **18,647 × 50,004**

## Models and Evaluation Metrics

| Algorithm | Key settings |
|---|---|
| Linear Regression | Default |
| Decision Tree Regressor | `max_depth=10`, `min_samples_split=10` |
| Random Forest Regressor | `n_estimators=100`, `max_depth=10`, `min_samples_split=10` |
| Gradient Boosting Regressor | `n_estimators=100`, `learning_rate=0.05`, `max_depth=3` |

| Metric | Meaning | Better when |
|---|---|---|
| MAE | Average size of prediction errors | Lower |
| MSE | Average of squared errors; penalises large mistakes | Lower |
| RMSE | Square root of MSE, in original units | Lower |
| R² | Share of target variation explained by the model | Higher |

## Results

Evaluated on the 18,647-row test set:

| Algorithm | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| **Linear Regression** | **0.0810** | **0.0124** | **0.1115** | **0.3301** |
| Gradient Boosting | 0.0847 | 0.0138 | 0.1176 | 0.2544 |
| Random Forest | 0.0850 | 0.0140 | 0.1182 | 0.2463 |
| Decision Tree | 0.0863 | 0.0144 | 0.1199 | 0.2248 |

## Best Model

**Linear Regression** gave the best results on every metric:

| MAE | MSE | RMSE | R² |
|---|---|---|---|
| 0.0810 | 0.0124 | 0.1115 | 0.3301 |

- It explains about 33% of the variation in title sentiment.
- Its average prediction error is about 0.08 on a scale from -1 to +1.
- A simple linear model outperformed the tree-based models, likely because it suits sparse, high-dimensional TF-IDF features.

## Conclusion

The project cleaned 93,235 news records, built 50,004 features from text and share counts, and compared four regression algorithms. Linear Regression performed best with an R² of 0.3301 and an RMSE of 0.1115.

## Future Scope

- Try regularised models such as Ridge and Lasso regression
- Add `Source` and `Topic` as features
- Use word embeddings or transformer-based models
- Tune hyperparameters with cross-validation

## Getting Started

### Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scikit-learn, scipy
- Jupyter Notebook

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

### Run

1. Download the dataset from the UCI Machine Learning Repository and save it as `news_popularity.csv` in the project folder.
2. Start Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open `Regression.ipynb` and run all cells.

## Repository Structure

```
.
├── Regression.ipynb        # Full analysis and modelling notebook
├── news_popularity.csv     # Dataset (download from UCI)
└── README.md
```

## Acknowledgements

- Dataset: UCI Machine Learning Repository
- Libraries: scikit-learn, pandas, NumPy, Matplotlib, Seaborn
