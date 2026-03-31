# Automated ICD Coding FNL Project

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![NLP](https://img.shields.io/badge/domain-NLP%20%7C%20Healthcare-green)

## Overview
This repository contains the codebase and reports for the Natural Language Processing (NLP) project on **Automated International Classification of Diseases (ICD) Coding**. The goal of this project is to build an end-to-end NLP pipeline to automatically assign hierarchical ICD codes to health-related documents such as Electronic Medical Records (EMRs). 

This project explores traditional (non-deep learning) baseline machine learning methods based on a thorough literature review, implementing feature extraction techniques like TF-IDF alongside classifiers like SVMs.

## Directory Structure
- `data/`: Contains datasets and ICD code mapping pairs (Ignored by Git).
- `docs/`: Holds project presentations, literature reviews, and final reports.
- `notebooks/`: Jupyter Notebooks for data exploration and evaluating non-deep learning baseline methods.
- `src/`: Python source code, including custom modules for preprocessing and evaluating multi-label text clinical data.

## Milestones
- **March 18th:** Project Introduction.
- **April 8th:** First Follow-up.
  - Literature review of reference papers (see `docs/literature_review.md`).
  - Analysis of datasets and challenges.
  - Proposal of machine learning methods.
- **May 11th:** Second Follow-up.
  - Implementation and evaluation of a first baseline method.
- **June 1st - June 3rd:** Final Presentation.
  - Final implementation.
  - Written report and oral presentation.

## Setup and Installation
1. Clone the repository.
   ```bash
   git clone https://github.com/user-name/FNL-Project.git
   cd FNL-Project
   ```
2. Create and activate a Virtual Environment.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```
4. Place testing datasets (`codification_data.csv`, `icd_d_p_pairs.csv`, etc.) in the `data/` directory.

## Authors
Group 10 FNL 2025-26 - 
Phoebe Iglesias, David Redrejo & Pau Rossell
