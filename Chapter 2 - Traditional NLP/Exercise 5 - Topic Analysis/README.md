# Chapter 2 – Exercise 5: Topic Analysis with NMF

[⬅ Back to the chapter](../README.md) · [🏠 Main page](../../README.md)

📄 **Exercise sheet:** [Exercise_5- Topic Analysis.docx](Exercise_5-%20Topic%20Analysis.docx)

## Goal

Discover the hidden topics of ~18,000 newsgroup posts with TF-IDF + Non-negative Matrix Factorization, show the 10 top words of each topic and compare the topics with the real categories.

## Key ideas (for a junior)

* **Unsupervised learning** – no labels are given: the algorithm finds groups of words that go together.
* **NMF** – TF-IDF matrix (posts × words) ≈ **W** (posts × topics) × **H** (topics × words). All values ≥ 0, so they read as "how much".
* **A topic = a row of H** – its top words describe it; *you* give it a name.
* **TF-IDF settings** – `max_df=0.95`, `min_df=2`, `max_features=5000`, letters only, English stopwords.

## What the script does

The script `ex5_topic_analysis_nmf.py` follows the exercise sheet step by step; every step is commented in the code.

| # | Task | How |
|---|---|---|
| 1 | Load and explore | posts per category, words per post, an example |
| 2 | TF-IDF |  |
| 3 | NMF with 20 topics | `NMF(n_components=20, init='nndsvda')` |
| 4 | 10 top words per topic | + the category that uses it most |

## How to run

In a terminal (Command Prompt, PowerShell or the VS Code terminal):

```bash
cd "Chapter 2 - Traditional NLP/Exercise 5 - Topic Analysis"
python ex5_topic_analysis_nmf.py --topics 20
```
You can also run it from any other folder: the script always finds its data and writes
into its own `outputs/` folder.  First time? Run `setup_environment.bat` in the main folder.

## Results

> **Graphs:** the 20 Newsgroups dataset is downloaded by scikit-learn the first time you run the script (about 14 MB, needs internet once). The graphs listed below then appear in `outputs/`.

## Common mistakes

* Keeping headers/signatures/quotes: NMF then finds "topics" made of e-mail addresses.
* Expecting topic *k* to be category *k*: topics come in no particular order and similar forums are often merged.

## Files

- `Exercise_5- Topic Analysis.docx` – the exercise sheet (from Moodle)
- `ex5_topic_analysis_nmf.py` – the solution, with comments
- `outputs/` – created by the script:
  - ex5_dataset_overview.png
  - ex5_top_words_per_topic.png
  - ex5_topic_vs_category.png
  - ex5_topics.csv
