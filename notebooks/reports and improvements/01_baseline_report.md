# Baseline Model Report: TF-IDF + LinearSVC

This document serves as a record of the current (v1.0) baseline model implementation for the single-label ICD-10 category classification task. It captures the exact parameters, preprocessing steps, and evaluation metrics obtained prior to applying any advanced improvements. 

Use this report as the ground-truth benchmark to evaluate whether future iterations (like Grid Search or Cross-Validation) actually improve the model.

---

## 1. Task Definition & Data Handling
* **Objective:** Single-label multi-class classification to predict the **first character** (category) of an ICD-10 code given a clinical literal.
* **Target Classes:** 36 unique categories (A-Z, 0-9).
* **Data Splitting:** 
  * 80% Training / 20% Validation.
  * **Stratified split** applied based on the target category to maintain class distribution.
  * *Training Samples:* 9,267
  * *Validation Samples:* 2,317
* **Conflict Resolution:** For literals mapping to multiple ICD codes with different starting characters, the **majority-vote category** was chosen to form a clean single-label dataset.

---

## 2. Preprocessing Steps (`normalize_text`)
Before feature extraction, every literal was passed through the following strict normalization pipeline:
1. Converted to lowercase.
2. HTML tags stripped (e.g., `<font>`).
3. **Accents completely stripped** using NFD unicode normalization (e.g., `á` → `a`, `ñ` → `n`).
4. Non-alphanumeric characters stripped (keeping only ASCII letters, digits, and spaces).
5. Multiple whitespaces collapsed into a single space.

---

## 3. Model Pipeline Parameters

### A. Feature Extraction (`TfidfVectorizer`)
* **Analyzer:** `char_wb` (character n-grams, respecting word boundaries).
* **N-gram Range:** `(3, 6)` (trigrams up to 6-grams).
* **Sublinear TF:** `True` (applies logarithmic scaling `1 + log(tf)` to dampen the impact of very frequent terms).
* **Max Features:** `100,000` (vocabulary capped).
* **Min Document Frequency (`min_df`):** `2` (ignores extremely rare n-grams).

### B. Classification Model (`LinearSVC`)
* **Algorithm:** Support Vector Machine with a linear kernel.
* **C (Regularization):** `1.0` (standard L2 penalty).
* **Class Weight:** `'balanced'` (automatically adjusts weights inversely proportional to class frequencies to handle severe imbalance).
* **Max Iterations:** `10,000`.
* **Random State:** `42`.

---

## 4. Evaluation Results (Validation Set)

The model was evaluated against the 20% hold-out validation set.

### Global Metrics
| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **0.5693** | The official metric. The model correctly predicts the exact category 56.9% of the time. |
| **Macro F1** | **0.5091** | Unweighted average across all 36 classes. Shows that minority classes are dragging down the overall average. |
| **Weighted F1** | **0.5658** | F1 score weighted by class support. Very close to accuracy. |
| **Weighted Precision** | **0.5759** | |
| **Weighted Recall** | **0.5693** | |

### Notable Per-Class Performance
* **Strong Performers:** 
  * Category `Z` (largest class): F1 = 0.76
  * Category `M`: F1 = 0.75
  * Category `B`: F1 = 0.75
* **Struggling Classes (Minority/Hard):**
  * Category `W`, `X`: F1 = 0.00 (Model failed to predict them entirely).
  * Category `7`: F1 = 0.18

---

## 5. Output / Submission
* **File Generated:** `submissions/svm_baseline.csv`
* **Leaderboard Literals Scored:** 6,667
* **Format:** Adhered strictly to `[id, Literal, y_category]` with **zero empty values** (all literals received a valid prediction out of the 36 possible categories).

---
*Generated prior to Improvement Phase 1.*
