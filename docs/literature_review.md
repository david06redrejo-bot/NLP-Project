# Literature Review: Automated ICD Coding
*(Referenced paper: A survey of automated International Classification of Diseases coding)*

## 1. Introduction
The International Classification of Diseases (ICD) provides a standardized way to code diagnoses. Assigning these codes automatically helps reduce human error, DRG payment issues, and the cost of maintaining professional coders.

## 2. Characteristics of ICD Codes
ICD codes form a hierarchical tree with inheritance (parent-child nodes), mutual exclusion (sibling nodes), and co-occurrence (friend nodes). This introduces unique challenges such as large label spaces and highly imbalanced data.

## 3. Machine Learning Approaches (Non-Deep Learning)
According to the literature review, early stages of automated coding relied on:
1. **Rule-based methods (e.g., if-else triggers based on ICD guidelines).**
2. **Traditional Machine Learning (Stage 2):**
   - **Models:** FlatSVM, hierarchy-based SVM, Random Forests, Naive Bayes.
   - **Features:** Term Frequency-Inverse Document Frequency (TF-IDF), Bag of Words (BoW), and n-grams from unstructured EMRs.

## 4. Proposed Methods
*(To be filled during the April 8th milestone)*
- Method 1: TF-IDF + FlatSVM.
- Method 2: N-gram extraction + Random Forest.
