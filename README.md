# Comment Category Prediction Challenge

## Overview
This project addresses the **Comment Category Prediction Challenge**, where the goal is to predict the final category assigned to each comment using a mix of textual, numerical, and categorical features.

The dataset contains:
- short user comments,
- engagement signals such as upvotes and downvotes,
- symbolic features like emoticons,
- internal system indicators,
- demographic/topic-related categorical fields.

The challenge focuses on learning patterns from noisy real-world data and building a strong multi-class classification pipeline.

---

## Problem Statement
Given a comment and its associated metadata, predict the **final label/category** assigned by the system.

This is a **multi-class classification** task evaluated using **Macro F1 Score**, which is especially useful for imbalanced datasets because it gives equal importance to each class.

---

## Dataset Description
The dataset includes both train and test files.

### Target
- `label` — final category for each comment

### Text Feature
- `comment` — raw comment text

### Numerical Features
- `emoticon_1`
- `emoticon_2`
- `emoticon_3`
- `if_1`
- `if_2`
- `upvote`
- `downvote`

### Categorical Features
- `race`
- `religion`
- `gender`
- `disability`
- `post_id`

---

## Project Workflow

### 1. Data Loading
Loaded train, test, and sample submission files into pandas DataFrames.

### 2. Data Cleaning
Performed basic preprocessing to handle missing values and noisy text:
- filled missing categorical values with `unknown`
- filled missing comments with empty strings
- removed irrelevant columns
- normalized comment text

### 3. Exploratory Data Analysis
Analyzed:
- class imbalance
- missing value patterns
- text length and punctuation behavior
- distribution of engagement-related features

### 4. Feature Engineering
Created custom features from the comment text and metadata:

#### Text-based features
- `uniq_ratio` — ratio of unique words to total words
- `caps_ratio` — proportion of uppercase characters
- `excl_count` — number of exclamation marks
- `avg_wlen` — average word length
- `word_count` — number of words
- `ques_count` — number of question marks
- `punc_ratio` — ratio of punctuation characters

#### Interaction-based features
- `down_up_ratio` — downvotes normalized by upvotes
- `total_votes` — combined vote count

#### Binned feature
- `if2_bin` — categorical binning of `if_2` into:
  - low
  - mid
  - high

### 5. Text Cleaning
Applied normalization to the comment text:
- lowercasing
- URL removal
- mention replacement
- hashtag normalization
- repeated character reduction
- special character filtering

### 6. Feature Extraction Pipeline
Used a combined preprocessing pipeline:
- **OneHotEncoder** for categorical variables
- **StandardScaler** for numerical variables
- **TF-IDF** for comment text using:
  - word n-grams `(1,3)`
  - character n-grams `(3,5)`

### 7. Model Experiments
Tried multiple models and compared their Macro F1 scores:
- Logistic Regression
- SGD Classifier
- Tuned SGD Classifier
- Logistic Regression + SVD
- Naive Bayes
- KNN + SVD
- Linear SVC
- Bagging with Logistic Regression
- LightGBM
- MLP + SVD
- Voting Ensemble

### 8. Final Model
The best performing model was a **Voting Classifier** combining:
- Bagging Logistic Regression
- Calibrated Linear SVC
- SGD Classifier

---

## Model Performance

| Model | Macro F1 |
|---|---:|
| Voting (SGD + BaggingLR + SVC) | 0.8295 |
| SGD (Tuned) | 0.8265 |
| Bagging (LR) | 0.8235 |
| LinearSVC | 0.8138 |
| SGD | 0.8078 |
| Logistic Regression | 0.8033 |
| MLP | 0.7460 |
| LightGBM | 0.7386 |
| LR + SVD | 0.7138 |
| Naive Bayes | 0.6197 |
| KNN | 0.5955 |

---

## Key Takeaways
- The dataset is **highly imbalanced**, so Macro F1 was the right evaluation metric.
- Text features carried strong predictive power.
- Combining **text TF-IDF**, **engineered metadata features**, and **ensemble modeling** produced the best results.
- The final voting ensemble gave the most robust generalization across classes.

---

## Repository Structure
```bash
.
├── notebook.ipynb
├── submission.csv
├── README.md
└── requirements.txt
