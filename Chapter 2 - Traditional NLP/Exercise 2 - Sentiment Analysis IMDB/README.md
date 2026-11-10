# Chapter 2 – Exercise 2: Sentiment Analysis with VADER on IMDB

[⬅ Back to the chapter](../README.md) · [🏠 Main page](../../README.md)

📄 **Exercise sheet:** [Exercise_2-Sentiment Analysis-IMDB.docx](Exercise_2-Sentiment%20Analysis-IMDB.docx)

## Goal

Run VADER on 50,000 IMDB movie reviews (after removing their HTML), measure its accuracy against the real labels, show the most positive/negative reviews, test several neutral thresholds and draw a pie chart.

## Key ideas (for a junior)

* **Ground truth** – the dataset already says if each review is positive or negative → we can *measure* VADER.
* **Accuracy** – correct predictions / all predictions. A random guess gets 50 %.
* **Confusion matrix** – true label (rows) vs predicted label (columns); the diagonal = correct.
* **IMDB has no neutral reviews** – every *neutral* prediction counts as wrong, so a wider neutral band lowers the accuracy (but the remaining decisions are more reliable).
* **Why VADER struggles here** – long reviews, plot descriptions full of negative words (*murder, war*), irony.

## What the script does

The script `ex2_sentiment_imdb_vader.py` follows the exercise sheet step by step; every step is commented in the code.

| # | Task | How |
|---|---|---|
| 1 | Load the CSV | from `../../data/` (csv or zip) |
| 2 | Remove HTML tags | regex `<[^>]+>` + `html.unescape` |
| 3 | VADER on every review |  |
| 4 | Classify (±0.05) |  |
| 5 | Write a CSV with compound + label |  |
| 6 | Compare with the true labels – accuracy | accuracy + confusion matrix |
| 7 | 10 most positive and 10 most negative | `nlargest` / `nsmallest` |
| 8 | Other neutral thresholds | 0 → 0.9, table + graph |
| 9 | Pie chart | VADER vs true labels |

## How to run

In a terminal (Command Prompt, PowerShell or the VS Code terminal):

```bash
cd "Chapter 2 - Traditional NLP/Exercise 2 - Sentiment Analysis IMDB"
python ex2_sentiment_imdb_vader.py --sample 5000
```
You can also run it from any other folder: the script always finds its data and writes
into its own `outputs/` folder.  First time? Run `setup_environment.bat` in the main folder.

Data: `EXERCISES/data/Ch2_Ex1_IMDB.csv.zip` (pandas reads the zip directly).

## Results

> **Graphs:** this exercise needs `NLTK (VADER)`.  Run the script once on your computer (after `setup_environment.bat`): the graphs listed below will appear in the `outputs/` folder.

The full run (50,000 reviews) takes a few minutes; use `--sample 5000` for a quick test.

## Common mistakes

* Forgetting the HTML: `<br />` tags are counted as words.
* Reading the CSV with a hand-made parser: `pd.read_csv` handles the quotes and commas inside the reviews.

## Files

- `Exercise_2-Sentiment Analysis-IMDB.docx` – the exercise sheet (from Moodle)
- `ex2_sentiment_imdb_vader.py` – the solution, with comments
- `outputs/` – created by the script:
  - Ch2_Ex2_IMDB_with_sentiment.csv (not on GitHub: 67 MB)
  - ex2_confusion_matrix.png
  - ex2_thresholds.png
  - ex2_pie_chart.png
  - ex2_compound_distribution.png
