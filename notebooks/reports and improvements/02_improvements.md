# Proposed Improvements for Baseline Model (TF-IDF + SVM)

Given the constraints that the core algorithmic pipeline (**TF-IDF for feature extraction** and **SVM for classification**) must remain untouched, there are still several powerful strategies to improve the model's performance on the ICD-10 category classification task. 

Currently, the baseline model achieves approximately **56.9% accuracy** and a **50.9% Macro-F1 score**. The following improvements focus on hyperparameter tuning, feature engineering, validation strategies, and data handling.

---

## 1. Hyperparameter Optimization (Grid Search)
Currently, the pipeline uses static, manually chosen parameters (e.g., `C=1.0` for SVM, `ngram_range=(3,6)` for TF-IDF). We can systematically search for the optimal parameters.

### Implementation Instructions:
1. **Import Search Tools**: In the notebook, import `GridSearchCV` or `RandomizedSearchCV` from `sklearn.model_selection`.
2. **Create a Pipeline**: Combine the `TfidfVectorizer` and `LinearSVC` into a single `sklearn.pipeline.Pipeline`.
3. **Define a Parameter Grid**:
   - **SVM `C` parameter**: Try `[0.1, 1.0, 10.0]`. A higher `C` fits the training data more strictly, while a lower `C` encourages a wider margin (better generalization).
   - **SVM `loss`**: Try `['hinge', 'squared_hinge']`.
   - **TF-IDF `min_df`**: Try `[1, 2, 5]` to see if filtering rare terms more aggressively reduces noise.
4. **Execute Search**: Run the search on the training set optimizing for `accuracy` or `f1_macro`.

### Expected Outcome:
Finding the optimal balance between bias and variance. It is expected that tuning the regularization parameter `C` will directly improve generalization on the validation set.

---

## 2. Advanced Feature Engineering (FeatureUnion)
Currently, the TF-IDF vectorizer only uses character n-grams (`char_wb`). While great for capturing typos and morphological roots (common in clinical text), it might miss the semantic weight of exact whole words.

### Implementation Instructions:
1. **Import FeatureUnion**: From `sklearn.pipeline` import `FeatureUnion`.
2. **Create Two Vectorizers**:
   - `char_tfidf`: The existing one with `analyzer='char_wb'`, `ngram_range=(3,6)`.
   - `word_tfidf`: A new one with `analyzer='word'`, `ngram_range=(1,2)` (unigrams and bigrams).
3. **Combine them**: Use `FeatureUnion([('char', char_tfidf), ('word', word_tfidf)])` to concatenate both feature spaces into a single massive sparse matrix.
4. **Train**: Pass this combined feature matrix into the SVM.

### Expected Outcome:
The model will learn from both structural/morphological sub-word patterns (from chars) and exact medical terminology combinations (from words), likely boosting accuracy and recall for complex literals.

---

## 3. Robust Evaluation with Stratified Cross-Validation
Currently, we use a single 80/20 train/validation split. This can result in high variance; the 56.9% accuracy might be artificially high or low depending on the random seed.

### Implementation Instructions:
1. **Import CV Tools**: Import `cross_validate` and `StratifiedKFold` from `sklearn.model_selection`.
2. **Merge Train/Val**: Instead of splitting the data manually, pass the entire dataset `cat_df` (Literals and Categories) into the cross-validator.
3. **Set Folds**: Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
4. **Evaluate**: Run `cross_validate` and compute the mean and standard deviation of the accuracy and macro-F1 across all 5 folds.

### Expected Outcome:
A more realistic, robust, and reliable metric of how the model will truly perform on the hidden test set (leaderboard). If the variance across folds is high, it indicates the model is sensitive to data distribution.

---

## 4. Preprocessing & Imbalance Handling Tweaks
The dataset is highly imbalanced (e.g., category `Z` has 1500+ samples, while `W` has only 7). While `class_weight='balanced'` helps the SVM, we can do more at the data level.

### Implementation Instructions:
1. **Stopword Removal**: Enhance `normalize_text` in `data_processing.py` to remove common Spanish stopwords (using `nltk.corpus.stopwords.words('spanish')`) which might be adding noise to the TF-IDF matrix.
2. **Oversampling (Optional)**: If you are open to adding new libraries, use `imblearn.over_sampling.SMOTE` to artificially generate synthetic TF-IDF vectors for minority classes (like `W`, `X`, `Y`) before passing them to the SVM.

### Expected Outcome:
Removing stopwords reduces the dimensionality and noise, allowing the SVM to focus on clinical terms. Addressing severe class imbalance should significantly raise the **Macro-F1** score by improving recall on minority classes.

---

## Next Steps: What to do with the new results?

Once you implement any of these improvements in the notebook:

1. **Compare against the Baseline**: Record the new `accuracy` and `macro_f1`. Did they exceed `0.5693` and `0.5091`, respectively? 
2. **Analyze the Classification Report**: Look specifically at the minority classes (e.g., `W`, `X`, `U`). Did the changes (like Cross-Validation or FeatureUnion) allow the model to finally predict them correctly? If precision/recall for these classes is still 0.00, it means the SVM requires more aggressive imbalance handling.
3. **Update the Final Submission**: If the cross-validated metrics are strictly better, retrain the `Pipeline` on the **entire** dataset (100% of `codification_data.csv`), transform the leaderboard data, and generate a new `svm_improved_submission.csv` to submit.
