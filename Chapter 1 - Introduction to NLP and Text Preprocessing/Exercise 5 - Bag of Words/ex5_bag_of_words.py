"""
Chapter 1 - Exercise 5 : Bag of Words (BoW)
============================================

KEY IDEAS FOR A JUNIOR
----------------------
A machine learning model only understands NUMBERS.  "Bag of Words" is the
simplest way to turn text into numbers:

    1. Build a VOCABULARY = the list of all different words in all documents.
    2. For every document, COUNT how many times each vocabulary word appears.

The result is the DOCUMENT-TERM MATRIX (DTM):

                 alice  rabbit  sister  watch ...
    doc1           1      0       2       0
    doc2           0      1       0       0
    doc3           1      2       0       2

    rows    = documents
    columns = words (the vocabulary)
    values  = how many times the word appears in the document

It is called a "bag" because the ORDER of the words is lost:
"dog bites man" and "man bites dog" give exactly the same row!

scikit-learn's CountVectorizer does steps 1 + 2 for us:
    fit()        -> learns the vocabulary
    transform()  -> counts the words
    fit_transform() = both at once

STOPWORDS (bonus): words like "the", "her", "of" are the most frequent words in
almost every English text, so they hide the interesting words.  With
CountVectorizer(stop_words="english") they are removed.

HOW TO RUN
----------
    python ex5_bag_of_words.py
    python ex5_bag_of_words.py myfile1.txt myfile2.txt   (your own documents)

OUTPUT
------
    * the DTM and the top 10 words in the terminal
    * outputs/ex5_dtm.csv                  (DTM with all words)
    * outputs/ex5_dtm_no_stopwords.csv
    * outputs/ex5_top_words.png            (top 10, with vs without stopwords)
    * outputs/ex5_dtm_heatmap.png          (the matrix as a picture)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

DEFAULT_FILES = ["Ch1_Ex5-6_doc1.txt", "Ch1_Ex5-6_doc2.txt", "Ch1_Ex5-6_doc3.txt"]

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)


# ---------------------------------------------------------------------------
# Step 1 - read the 3 documents from disk
# ---------------------------------------------------------------------------
def read_documents(file_names):
    """Return (names, texts).  Files are searched next to this script."""
    names, texts = [], []
    for name in file_names:
        path = Path(name) if Path(name).is_absolute() else HERE / name
        if not path.exists():
            sys.exit(f"File not found: {path}")
        # "with" closes the file automatically, encoding avoids problems with é, ’ ...
        with open(path, encoding="utf-8") as f:
            texts.append(f.read())
        names.append(path.stem.replace("Ch1_Ex5-6_", ""))   # "doc1", "doc2", ...
    return names, texts


def build_dtm(texts, names, stop_words=None):
    """Step 2 + 3: CountVectorizer -> DTM -> pandas DataFrame."""
    vectorizer = CountVectorizer(stop_words=stop_words)   # lowercase=True by default
    matrix = vectorizer.fit_transform(texts)               # a *sparse* matrix (mostly zeros)
    dtm = pd.DataFrame(
        matrix.toarray(),                                   # sparse -> normal table
        index=names,                                        # rows    = documents
        columns=vectorizer.get_feature_names_out(),         # columns = words
    )
    return dtm


def top_words(dtm, n=10):
    """Step 4: sum every column = total count of the word in ALL documents."""
    return dtm.sum(axis=0).sort_values(ascending=False).head(n)


def main():
    files = sys.argv[1:] or DEFAULT_FILES
    names, texts = read_documents(files)

    print("=" * 80)
    print("STEP 1 - The documents")
    print("=" * 80)
    for n, t in zip(names, texts):
        print(f"\n[{n}] {' '.join(t.split())}")

    # Steps 2 + 3 --------------------------------------------------------------
    dtm = build_dtm(texts, names)
    print("\n" + "=" * 80)
    print(f"STEPS 2-3 - Document-Term Matrix: {dtm.shape[0]} documents x {dtm.shape[1]} words")
    print("=" * 80)
    print(dtm.to_string())
    dtm.to_csv(OUT / "ex5_dtm.csv")

    # Step 4 ----------------------------------------------------------------
    top_all = top_words(dtm)
    print("\n" + "=" * 80)
    print("STEP 4 - Top 10 most frequent words (all documents together)")
    print("=" * 80)
    print(top_all.to_string())

    # Bonus ------------------------------------------------------------------
    dtm_ns = build_dtm(texts, names, stop_words="english")
    dtm_ns.to_csv(OUT / "ex5_dtm_no_stopwords.csv")
    top_ns = top_words(dtm_ns)
    print("\n" + "=" * 80)
    print("BONUS - Top 10 WITHOUT stopwords")
    print("=" * 80)
    print(top_ns.to_string())
    print(f"\nVocabulary size: {dtm.shape[1]} words with stopwords -> {dtm_ns.shape[1]} without.")
    lost = [w for w in top_all.index if w not in dtm_ns.columns]
    print(f"Stopwords that were in the top 10: {lost}")
    print("Conclusion: with stopwords the top 10 is full of 'grammar' words (her, the, of...);")
    print("            without them we see what the texts are ABOUT (alice, rabbit, sister...).")

    # ------------------------------------------------------------------------
    # GRAPHS
    # ------------------------------------------------------------------------
    BLUE, GREY = "#2a78d6", "#52514e"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=False)
    for ax, series, ttl in [
        (axes[0], top_all, "Top 10 words - WITH stopwords"),
        (axes[1], top_ns, "Top 10 words - WITHOUT stopwords"),
    ]:
        ax.barh(series.index, series.values, color=BLUE if ax is axes[1] else GREY, height=0.6)
        for y, v in enumerate(series.values):
            ax.text(v + 0.05, y, str(v), va="center", color=GREY, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Total count in the 3 documents")
        ax.set_title(ttl, loc="left", weight="bold")
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e5e4e0", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Chapter 1 - Exercise 5: Bag of Words", weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "ex5_top_words.png", dpi=150)
    plt.close(fig)

    # Heatmap of the DTM without stopwords (only the 20 most frequent words so it stays readable)
    cols = top_words(dtm_ns, 20).index
    sub = dtm_ns[cols]
    fig, ax = plt.subplots(figsize=(13, 3.6))
    im = ax.imshow(sub.values, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(sub.index)), sub.index)
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            v = sub.values[i, j]
            ax.text(j, i, v, ha="center", va="center", fontsize=9,
                    color="white" if v > sub.values.max() / 2 else "#0b0b0b")
    ax.set_title("Document-Term Matrix (20 most frequent words, stopwords removed) - each cell = a count",
                 loc="left", weight="bold")
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(OUT / "ex5_dtm_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"\nFiles saved in: {OUT}")


if __name__ == "__main__":
    main()
