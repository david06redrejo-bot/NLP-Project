# Presentation Guide — April 8th Follow-up

> **Slides file:** `docs/presentation_april8.html` — Open in Chrome/Edge, press **F** for fullscreen.
> **Duration:** 2–3 minutes · 2 slides

---

## What the teacher is evaluating

The teacher wants to see three things in this short session:

1. **You understand the task.** Not just "it's NLP" — specifically what makes *this* dataset unusual and what that implies for your method choice.
2. **You read the literature.** The survey paper defines three historical stages of ICD coding. Your proposal should land explicitly in one of them and name the researchers behind it.
3. **You have a concrete plan.** A method with a justification, not "we'll try some things."

---

## Slide 1 — Task & Dataset Analysis

This slide answers the question: *what is the problem and why is it hard?*

### The task
Automated ICD coding means taking a short piece of unstructured text written by a clinician and predicting which ICD-10 codes it corresponds to. This is fundamentally a **text classification** problem, but with two complicating factors that make it harder than a typical sentiment analysis or topic classifier:

- **Multi-label:** A single input can have multiple correct outputs simultaneously. The word "parto" (birth) in a clinical note can co-occur with an obstetric condition code, a blood group code, and a gestational age code all at once — up to 14 codes for one phrase in our training data.
- **Open label space:** Unlike most classification benchmarks where you have 10 or 100 fixed classes, here there are 4,059 unique ICD codes in the training set alone, and ~179,000 in the full dictionary. The classifier must choose from a very large number of possible outputs.

### The key insight from the dataset
The two highlighted statistics on the right column are the anchor of the entire presentation:

- **2.2 words average:** Our clinical literals are extremely short. "HTA parto", "Alergia ibuprofeno", "miocardiopatía dilatada" — these are not paragraphs, they are fragments. This is completely different from the MIMIC dataset (the English benchmark most papers use), where discharge summaries average 1,500 tokens. This has a direct consequence for method choice: deep learning models that learn contextual representations across long sequences have no context to exploit here. A bag of words or TF-IDF vector captures essentially the same information as a transformer on 2-word inputs.

- **47% of codes appear only once:** Nearly half of the 4,059 unique codes have exactly one training example. A supervised classifier fundamentally cannot generalise from a single positive example. This is the long-tail problem. It means that even a perfect model will fail on a large share of the label space simply due to data scarcity — and it means the training data is extremely unbalanced.

Together, these two facts — short text + extreme label imbalance — define the entire challenge and justify the method choices on slide 2.

### The language and abbreviation challenge
Spanish clinical text adds a layer of noise that English NLP pipelines don't handle: the data mixes Spanish and Catalan (e.g. "parto" / "part", "migraña" / "migranya"), uses heavy medical abbreviations (HTA = hypertension, ITU = urinary tract infection, IRC = chronic renal failure), and has 11.8% of literals written entirely in ALL CAPS. Standard tokenisers and pre-trained Spanish language models weren't trained on this register.

---

## Slide 2 — Proposed Solution

This slide answers the question: *what are we going to build and why?*

### Why Traditional Machine Learning (Stage 2)

The survey paper organises the history of automated ICD coding into three stages:

- **Stage 1 — Rule-based:** handcrafted if-else logic from coding guidelines. Breaks down at scale (thousands of codes).
- **Stage 2 — Traditional ML:** TF-IDF features + supervised classifiers (SVM, Naive Bayes, Logistic Regression). Scalable but ignores inter-code relationships.
- **Stage 3 — Deep Learning:** neural representations (CNNs, LSTMs, Transformers). Best results at scale, but requires long documents and large training data.

Given our 2.2-word average, Stage 2 is the appropriate entry point. This is not just a default — it is a deliberate, literature-backed argument.

### The pipeline
The five-step pipeline on the slide reflects a standard supervised NLP workflow:

1. **Input text** — the raw Spanish/Catalan clinical phrase.
2. **Preprocessing** — lowercasing, accent normalisation, removing HTML artifacts (some literals in the training data contain raw HTML tags), handling abbreviations.
3. **Feature extraction** — converting text into numerical vectors. This is the step where the three methods differ.
4. **Classification** — a One-vs-Rest wrapper trains one binary classifier per ICD code. The output is a set of positive predictions.
5. **Output** — one or more ICD-10 codes for each input literal.

### The three baselines and their citations

The three method cards correspond directly to researchers cited in Section 4.2 of the survey:

**TF-IDF + FlatSVM (Perotte et al. [32])**
TF-IDF weights each word by how informative it is across the corpus — frequent words across all documents get low weight, rare discriminative words get high weight. The result is a sparse high-dimensional vector for each literal. A Linear SVM then finds a hyperplane that separates positive from negative examples for each code independently. "Flat" means no hierarchical structure is exploited — each of the 4,059 codes is treated as its own independent binary problem. This is the gold standard baseline from the literature.

**Char N-gram + SVM (Marafino et al. [37])**
Instead of operating on whole words, this method breaks every word into overlapping character sequences of length 3 to 5. For example, "migraña" produces n-grams like "mig", "igr", "gra", "rañ", etc. This has two advantages specific to our data: (1) abbreviations share character-level patterns even when written differently (HTA vs hta vs H.T.A.), and (2) Spanish/Catalan cognates share character roots even when the full words differ. It is more robust to the noise in clinical text than word-level TF-IDF.

**Bag-of-Words + Naive Bayes (Elyne et al. [16])**
The simplest possible text representation: a vector counting how often each word appears, with no weighting. Complement Naive Bayes (a variant designed for class-imbalanced data) then estimates the probability of each code given the word counts using Bayes' theorem. It trains in seconds even on 13,700 examples. Its role is to establish the performance floor — if TF-IDF + SVM doesn't clearly outperform Naive Bayes, that reveals a problem with the feature engineering or the SVM configuration.

### The two acknowledged limitations

These appear at the bottom of slide 2. Mentioning them proactively tells the teacher you understand the weaknesses of your own proposal — which is more impressive than pretending there are none.

**Computational impracticality:** Training 4,059 individual binary SVMs is expensive. The survey itself notes this problem. The mitigation is a minimum-frequency filter: only train classifiers for codes that appear at least N times in training, where N is chosen so each classifier has enough positive examples to generalise. The exact threshold is an empirical decision we will determine during implementation.

**Code dependency ignored:** Flat One-vs-Rest models treat each ICD code as completely independent. In reality, certain codes almost always co-occur (e.g. obstetric codes tend to cluster), and models that capture these correlations perform better. This is what Stage 3 (deep learning) addresses — graph-based models and attention mechanisms can learn which codes are semantically related. Our baseline does not do this, but it gives us the benchmark performance that future deep learning work will need to beat.

---

## Evaluation strategy

- **Micro-F1** is the primary metric. It aggregates precision and recall across all (literal, code) pairs, weighted by frequency. A high Micro-F1 means the model is doing well on the common codes.
- **Macro-F1** averages F1 per code, giving equal weight to rare and frequent codes. This number will likely be much lower because of the long-tail. Reporting both shows the teacher you understand the difference between average-case and worst-case performance.

---

## How to open and present

```
1. Open Chrome or Edge
2. Navigate to: docs/presentation_april8.html
3. Click "Fullscreen" button at the bottom, or press F
4. To export as PDF: click "Print / Export PDF" → set layout to Landscape → Save as PDF
```

---

## Anticipated questions

| Question | The concept behind the answer |
|---|---|
| *Why not BERT or a transformer?* | Transformers learn from context across long sequences. Our median literal is 2 words — there is no context to attend to. TF-IDF is the appropriate tool for this data regime. |
| *How do you handle codes that appear only once?* | You filter them out for the baseline. Training a binary classifier requires at least some positive examples. Rare-code prediction is a few-shot or zero-shot problem, which is a separate research question. |
| *Which paper are these methods from?* | Survey Section 4.2, Table 1 — Perotte [32], Marafino [37], Elyne [16]. |
| *What performance do you expect?* | Perotte reports Micro-F1 ≈ 0.29 on MIMIC-III. Our dataset is shorter and in Spanish, so the results will differ — that is exactly what this baseline is meant to measure. |
