"""
Chapter 2 - Exercise 1 : Sentiment Analysis with VADER (product reviews)
=========================================================================

KEY IDEAS FOR A JUNIOR
----------------------
SENTIMENT ANALYSIS = deciding if a text is positive, negative or neutral.

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a RULE-BASED
method: no training, no data needed.  It contains a dictionary of ~7500 words,
each with a score ("love" = +3.2, "worst" = -3.1, "okay" = +0.9 ...), plus
rules for:
    * negation        "not good"     -> the score flips
    * intensifiers    "very good"    -> stronger
    * "but"           "nice, but slow" -> the part after "but" counts more
    * CAPITALS / !!!  "GREAT!!!"     -> stronger
    * emojis / slang  ":)"  "lol"

polarity_scores(text) returns 4 numbers:
    neg, neu, pos  -> proportion of negative / neutral / positive words (sum = 1)
    compound       -> ONE global score between -1 (very negative) and +1 (very positive)

Rule of the exercise (from the VADER authors):
    compound >= 0.05  -> positive
    compound <= -0.05 -> negative
    otherwise         -> neutral

SETUP (once)
------------
    pip install nltk pandas matplotlib
(the VADER dictionary is downloaded automatically the first time)

HOW TO RUN
----------
    python ex1_sentiment_vader.py
    python ex1_sentiment_vader.py --file other_reviews.csv --column review

OUTPUT
------
    * outputs/Ch2_Ex1_reviews_with_sentiment.csv   (review + compound + label)
    * outputs/ex1_compound_scores.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
except ImportError:
    sys.exit("NLTK is not installed.  Run:  pip install nltk")

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

# Colours: one per class, always the same in every Chapter 2 graph
COLORS = {"positive": "#1baf7a", "neutral": "#9a9890", "negative": "#e34948"}


def get_vader():
    """Return a VADER analyser, downloading the dictionary if needed."""
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        print("Downloading the VADER lexicon (only the first time)...")
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


def label_from_compound(compound, threshold=0.05):
    """Step 3 - turn the compound score into a label."""
    if compound >= threshold:
        return "positive"
    if compound <= -threshold:
        return "negative"
    return "neutral"


def main():
    parser = argparse.ArgumentParser(description="VADER sentiment analysis on a CSV of reviews")
    parser.add_argument("--file", default="Ch2_Ex1_rewiews.csv", help="CSV file (next to this script)")
    parser.add_argument("--column", default="review", help="name of the column with the text")
    args = parser.parse_args()

    # Step 1 - load the CSV ------------------------------------------------------
    path = Path(args.file) if Path(args.file).is_absolute() else HERE / args.file
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} reviews from {path.name}")

    # Step 2 - VADER on every review -------------------------------------------------
    sia = get_vader()
    scores = df[args.column].astype(str).apply(sia.polarity_scores)   # one dict per row
    scores = pd.DataFrame(scores.tolist())                             # -> 4 columns
    df = pd.concat([df, scores], axis=1)

    # Step 3 - classify ------------------------------------------------------------------
    df["sentiment"] = df["compound"].apply(label_from_compound)

    # Step 4 - write the new CSV ----------------------------------------------------------
    out_csv = OUT / "Ch2_Ex1_reviews_with_sentiment.csv"
    df[[args.column, "compound", "sentiment", "neg", "neu", "pos"]].to_csv(out_csv, index=False)

    print("\n" + "=" * 90)
    print(f"{'COMPOUND':>9}  {'LABEL':<9} REVIEW")
    print("=" * 90)
    for _, r in df.sort_values("compound", ascending=False).iterrows():
        print(f"{r['compound']:>+9.3f}  {r['sentiment']:<9} {r[args.column]}")

    print("\nNumber of reviews per label:")
    print(df["sentiment"].value_counts().to_string())
    print("\nHow to read it: VADER only knows WORDS.  A review such as 'It's a phone' contains")
    print("no emotional word, so VADER can only call it neutral.  Mixed reviews ('beautiful,")
    print("but slow') depend on which side has the stronger words: check them in the table!")

    # ------------------------------------------------------------------
    # GRAPH: compound score of every review (sorted), threshold band shaded
    # ------------------------------------------------------------------
    d = df.sort_values("compound")
    short = d[args.column].str.slice(0, 45) + d[args.column].apply(lambda t: "..." if len(t) > 45 else "")
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(d) + 1.5))
    ax.barh(short, d["compound"], color=[COLORS[s] for s in d["sentiment"]], height=0.6)
    ax.axvspan(-0.05, 0.05, color="#e5e4e0", zorder=0, label="neutral band (-0.05 ... +0.05)")
    ax.axvline(0, color="#52514e", linewidth=0.8)
    for y, (v, lab) in enumerate(zip(d["compound"], d["sentiment"])):
        ax.text(v + (0.03 if v >= 0 else -0.03), y, f"{v:+.2f} {lab}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color="#52514e")
    ax.set_xlim(-1.25, 1.25)
    ax.set_xlabel("VADER compound score  (-1 = very negative, +1 = very positive)")
    ax.set_title("Chapter 2 - Exercise 1: sentiment of each review", loc="left", weight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "ex1_compound_scores.png", dpi=150)
    plt.close(fig)

    print(f"\nCSV saved to  : {out_csv}")
    print(f"Graph saved to: {OUT / 'ex1_compound_scores.png'}")


if __name__ == "__main__":
    main()
