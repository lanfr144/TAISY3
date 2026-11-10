"""
Chapter 2 - Exercise 2 : Sentiment Analysis with VADER on IMDB movie reviews
=============================================================================

WHAT IS NEW COMPARED TO EXERCISE 1?
-----------------------------------
1. The data is BIG: 50 000 real movie reviews.
2. The reviews contain HTML ("<br /><br />") -> must be cleaned first.
3. The file already contains the TRUE answer (column "sentiment"), written
   by the people themselves.  So we can MEASURE how good VADER is:

        ACCURACY = number of correct predictions / number of reviews

KEY IDEAS FOR A JUNIOR
----------------------
* CONFUSION MATRIX - a table "what was true" (rows) vs "what VADER said"
  (columns).  The diagonal = correct answers, everything else = mistakes.
* IMDB labels are only "positive" / "negative".  Every review VADER calls
  "neutral" is therefore automatically WRONG.  That is why the size of the
  neutral band (the threshold) changes the accuracy - task 8.
* VADER was designed for SHORT social-media texts.  Movie reviews are long,
  often describe a sad or violent STORY (negative words!) while the reviewer
  liked the film, and use irony.  So do not expect a perfect score: compare
  the result with the 50 % of a random guess, and with the trained models of
  Chapter 2 (Naive Bayes) and Chapter 3 (neural networks).

HOW TO RUN
----------
    python ex2_sentiment_imdb_vader.py                 (all 50 000 reviews, a few minutes)
    python ex2_sentiment_imdb_vader.py --sample 5000   (quick test on 5 000 reviews)

The dataset is read from  ../../data/Ch2_Ex1_IMDB.csv  (or the .zip version).

OUTPUT (in outputs/)
--------------------
    Ch2_Ex2_IMDB_with_sentiment.csv  task 5
    ex2_confusion_matrix.png         task 6
    ex2_thresholds.png               task 8
    ex2_pie_chart.png                task 9
    ex2_compound_distribution.png    extra: where do the mistakes come from?
"""

import argparse
import html
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
except ImportError:
    sys.exit("NLTK is not installed.  Run:  pip install nltk")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                     # .../EXERCISES
DATA_CANDIDATES = [ROOT / "data" / "Ch2_Ex1_IMDB.csv", ROOT / "data" / "Ch2_Ex1_IMDB.csv.zip"]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

LABELS = ["positive", "neutral", "negative"]
COLORS = {"positive": "#1baf7a", "neutral": "#9a9890", "negative": "#e34948"}
GREY = "#52514e"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_vader():
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        print("Downloading the VADER lexicon (only the first time)...")
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


def load_imdb(sample=None):
    """Task 1 - read the CSV (pandas can read a .zip that contains one CSV)."""
    for path in DATA_CANDIDATES:
        if path.exists():
            df = pd.read_csv(path)
            print(f"Loaded {len(df):,} reviews from {path.relative_to(ROOT)}")
            break
    else:
        sys.exit("IMDB dataset not found.  Put Ch2_Ex1_IMDB.csv (or .csv.zip) in EXERCISES/data/")
    if sample:
        # random_state = fixed seed -> the same sample every time (reproducible)
        df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)
        print(f"Using a random sample of {len(df):,} reviews")
    return df


TAG = re.compile(r"<[^>]+>")       # "<" + anything except ">" + ">"   e.g. <br />


def clean_html(text):
    """Task 2 - remove HTML tags and decode HTML entities.

    "Great!<br /><br />Really &amp; truly" -> "Great! Really & truly"
    """
    text = TAG.sub(" ", text)          # remove the tags
    text = html.unescape(text)         # &amp; -> &   &quot; -> "
    return re.sub(r"\s+", " ", text).strip()


def classify(compound, threshold):
    """Task 4 - compound score -> label, with a neutral band [-threshold, +threshold]."""
    return np.where(compound >= threshold, "positive",
                    np.where(compound <= -threshold, "negative", "neutral"))


def style(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#e5e4e0", linewidth=0.8)
        ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="VADER on the IMDB dataset")
    parser.add_argument("--sample", type=int, default=None, help="only use N random reviews")
    args = parser.parse_args()

    df = load_imdb(args.sample)

    # Task 2 - remove HTML ------------------------------------------------------
    before = df["review"].str.count(r"<[^>]+>").sum()
    df["clean_review"] = df["review"].astype(str).apply(clean_html)
    print(f"Task 2: removed {before:,} HTML tags")

    # Task 3 - VADER ----------------------------------------------------------
    sia = get_vader()
    print(f"Task 3: running VADER on {len(df):,} reviews (be patient)...")
    t0 = time.time()
    compounds = []
    for i, text in enumerate(df["clean_review"], start=1):
        compounds.append(sia.polarity_scores(text)["compound"])
        if i % 10_000 == 0:
            print(f"   {i:,} done ({time.time() - t0:.0f}s)")
    df["compound"] = compounds
    print(f"   finished in {time.time() - t0:.0f}s")

    # Task 4 - classify with the standard threshold 0.05 -----------------------
    df["vader_sentiment"] = classify(df["compound"].to_numpy(), 0.05)

    # Task 5 - write the CSV ----------------------------------------------------
    out_csv = OUT / "Ch2_Ex2_IMDB_with_sentiment.csv"
    df[["clean_review", "sentiment", "compound", "vader_sentiment"]].rename(
        columns={"clean_review": "review", "sentiment": "true_sentiment"}
    ).to_csv(out_csv, index=False)
    print(f"Task 5: saved {out_csv.name}")

    # Task 6 - compare with the true labels -------------------------------------
    correct = df["vader_sentiment"] == df["sentiment"]
    accuracy = correct.mean()
    not_neutral = df["vader_sentiment"] != "neutral"
    acc_decided = correct[not_neutral].mean()
    print("\n" + "=" * 80)
    print("TASK 6 - HOW GOOD IS VADER?")
    print("=" * 80)
    print(f"  Accuracy (neutral counted as wrong)      : {accuracy:6.2%}")
    print(f"  Accuracy on the reviews VADER decided on : {acc_decided:6.2%}"
          f"  ({not_neutral.mean():.1%} of the reviews)")
    print(f"  A random guess would get about           :  50.00%")

    cm = pd.crosstab(df["sentiment"], df["vader_sentiment"]).reindex(
        index=["positive", "negative"], columns=LABELS, fill_value=0)
    print("\n  Confusion matrix (rows = TRUE label, columns = VADER):")
    print(cm.to_string().replace("\n", "\n  "))
    tp_rate = cm.loc["positive", "positive"] / cm.loc["positive"].sum()
    tn_rate = cm.loc["negative", "negative"] / cm.loc["negative"].sum()
    print(f"\n  Positive reviews recognised: {tp_rate:.1%}    Negative reviews recognised: {tn_rate:.1%}")
    if tp_rate > tn_rate:
        print("  -> VADER is better at finding POSITIVE reviews.  Many negative reviews are")
        print("     written politely or describe the plot, so VADER sees positive words.")
    else:
        print("  -> VADER is better at finding NEGATIVE reviews here: positive reviews that")
        print("     tell a dark story (crime, war, horror) contain many negative words.")

    # Task 7 - top 10 positive / negative ---------------------------------------
    print("\n" + "=" * 80)
    print("TASK 7 - THE 10 MOST POSITIVE AND THE 10 MOST NEGATIVE REVIEWS")
    print("=" * 80)
    for name, part in [("MOST POSITIVE", df.nlargest(10, "compound")),
                       ("MOST NEGATIVE", df.nsmallest(10, "compound"))]:
        print(f"\n--- {name} ---")
        for rank, (_, r) in enumerate(part.iterrows(), start=1):
            ok = "OK " if r["vader_sentiment"] == r["sentiment"] else "BAD"
            print(f"{rank:2d}. [{r['compound']:+.4f}] [true: {r['sentiment']:<8}] {ok} "
                  f"{r['clean_review'][:110]}...")
    print("\nNote: very long reviews get extreme scores easily (many words add up).")

    # Task 8 - try other neutral thresholds --------------------------------------
    print("\n" + "=" * 80)
    print("TASK 8 - DIFFERENT THRESHOLDS FOR 'NEUTRAL'")
    print("=" * 80)
    thresholds = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    rows = []
    for t in thresholds:
        pred = classify(df["compound"].to_numpy(), t)
        rows.append({
            "threshold": t,
            "accuracy": (pred == df["sentiment"].to_numpy()).mean(),
            "neutral_share": (pred == "neutral").mean(),
            "accuracy_decided": (pred == df["sentiment"].to_numpy())[pred != "neutral"].mean(),
        })
    res = pd.DataFrame(rows)
    print(res.to_string(index=False, formatters={
        "accuracy": "{:.2%}".format, "neutral_share": "{:.2%}".format, "accuracy_decided": "{:.2%}".format}))
    best = res.loc[res["accuracy"].idxmax()]
    print(f"\n  Best overall accuracy with threshold = {best['threshold']} ({best['accuracy']:.2%}).")
    print("  * A BIGGER neutral band -> more reviews are 'neutral' -> counted wrong ->")
    print("    the overall accuracy goes DOWN...")
    print("  * ...but the reviews VADER still decides on are more reliable (last column).")
    print(f"  * IMDB has no neutral reviews, so here the best band is T = {best['threshold']}.")

    # Task 9 - pie chart ----------------------------------------------------------
    counts = df["vader_sentiment"].value_counts().reindex(LABELS, fill_value=0)
    true_counts = df["sentiment"].value_counts().reindex(LABELS, fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, c, ttl in [(axes[0], counts, "VADER prediction (threshold 0.05)"),
                       (axes[1], true_counts, "True labels in the dataset")]:
        c = c[c > 0]
        ax.pie(c.values, labels=[f"{k}\n{v:,}" for k, v in c.items()],
               colors=[COLORS[k] for k in c.index], autopct="%1.1f%%", startangle=90,
               counterclock=False, wedgeprops={"edgecolor": "white", "linewidth": 2},
               textprops={"color": "#0b0b0b"})
        ax.set_title(ttl, weight="bold")
    fig.suptitle(f"Chapter 2 - Exercise 2: share of positive / neutral / negative ({len(df):,} reviews)",
                 weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "ex2_pie_chart.png", dpi=150)
    plt.close(fig)

    # Graph - confusion matrix ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(3), [f"VADER: {c}" for c in LABELS])
    ax.set_yticks(range(2), [f"TRUE: {r}" for r in cm.index])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm.values[i, j]
            ax.text(j, i, f"{v:,}\n({v / cm.values[i].sum():.0%} of row)", ha="center", va="center",
                    color="white" if v > cm.values.max() / 2 else "#0b0b0b", fontsize=10)
    ax.set_title(f"Confusion matrix - accuracy {accuracy:.1%}", loc="left", weight="bold")
    fig.colorbar(im, ax=ax, label="number of reviews")
    fig.tight_layout()
    fig.savefig(OUT / "ex2_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Graph - thresholds ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(res["threshold"], res["accuracy"] * 100, marker="o", linewidth=2, color="#2a78d6",
            label="accuracy (neutral = wrong)")
    ax.plot(res["threshold"], res["accuracy_decided"] * 100, marker="o", linewidth=2, color="#eb6834",
            label="accuracy on decided reviews")
    ax.plot(res["threshold"], res["neutral_share"] * 100, marker="o", linewidth=2, color="#9a9890",
            label="% of reviews called neutral")
    for _, r in res.iterrows():
        ax.annotate(f"{r['accuracy']:.0%}", (r["threshold"], r["accuracy"] * 100),
                    xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8, color=GREY)
    ax.set_xlabel("Neutral threshold  T   (neutral if  -T < compound < +T)")
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 100)
    ax.set_title("Task 8: a wider neutral band lowers the overall accuracy", loc="left", weight="bold")
    ax.legend(frameon=False)
    style(ax, "both")
    fig.tight_layout()
    fig.savefig(OUT / "ex2_thresholds.png", dpi=150)
    plt.close(fig)

    # Graph - distribution of compound by true label -----------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(-1, 1, 41)
    for lab in ["positive", "negative"]:
        ax.hist(df.loc[df["sentiment"] == lab, "compound"], bins=bins, alpha=0.6,
                color=COLORS[lab], label=f"true {lab}", edgecolor="white")
    ax.axvspan(-0.05, 0.05, color="#e5e4e0", zorder=0, label="neutral band")
    ax.set_xlabel("VADER compound score")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Compound score of truly positive vs truly negative reviews", loc="left", weight="bold")
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "ex2_compound_distribution.png", dpi=150)
    plt.close(fig)

    print(f"\nAll files saved in: {OUT}")


if __name__ == "__main__":
    main()
