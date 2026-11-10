# Chapter 2 – Exercise 1: Sentiment Analysis with VADER

[⬅ Back to the chapter](../README.md) · [🏠 Main page](../../README.md)

📄 **Exercise sheet:** [Exercise_1-Sentiment Analysis.docx](Exercise_1-Sentiment%20Analysis.docx)

## Goal

Score 11 product reviews with VADER, label each one positive / negative / neutral, and save the reviews with their compound score and label in a new CSV.

## Key ideas (for a junior)

* **Sentiment analysis** – is a text positive, negative or neutral?
* **VADER** – a *rule-based* method: a dictionary of ~7,500 scored words + rules for negation (*not good*), intensifiers (*very*), *but*, CAPITALS, `!!!` and emojis. No training needed.
* **`polarity_scores()`** – `neg`, `neu`, `pos` (proportions) and **`compound`** (one score from −1 to +1).
* **Thresholds** – compound ≥ 0.05 → positive, ≤ −0.05 → negative, otherwise neutral.

## What the script does

The script `ex1_sentiment_vader.py` follows the exercise sheet step by step; every step is commented in the code.

| # | Task | How |
|---|---|---|
| 1 | Load the CSV | `pd.read_csv` |
| 2 | VADER on each review | `sia.polarity_scores` |
| 3 | Classify with the compound score | `label_from_compound` |
| 4 | Write a CSV with compound + label | `outputs/Ch2_Ex1_reviews_with_sentiment.csv` |

## How to run

In a terminal (Command Prompt, PowerShell or the VS Code terminal):

```bash
cd "Chapter 2 - Traditional NLP/Exercise 1 - Sentiment Analysis"
python ex1_sentiment_vader.py
```
You can also run it from any other folder: the script always finds its data and writes
into its own `outputs/` folder.  First time? Run `setup_environment.bat` in the main folder.

Needs: `pip install nltk` (the VADER dictionary is downloaded automatically).

## Results

> **Graphs:** this exercise needs `NLTK (VADER)`.  Run the script once on your computer (after `setup_environment.bat`): the graphs listed below will appear in the `outputs/` folder.

## Common mistakes

* Comparing the whole dictionary `scores >= 0.05` instead of `scores["compound"]` → `TypeError`.
* Forgetting `nltk.download("vader_lexicon")` the first time.

## Files

- `Exercise_1-Sentiment Analysis.docx` – the exercise sheet (from Moodle)
- `ex1_sentiment_vader.py` – the solution, with comments
- `outputs/` – created by the script:
  - Ch2_Ex1_reviews_with_sentiment.csv
  - ex1_compound_scores.png – the compound score of each review, neutral band shaded
