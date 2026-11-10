"""
Chapter 1 - Exercise 6 : TF-IDF
================================

KEY IDEAS FOR A JUNIOR
----------------------
Bag of Words (Exercise 5) only COUNTS words.  Problem: "the" and "her" win
every time, although they say nothing about the document.

TF-IDF gives every word a WEIGHT instead of a count:

    TF  (Term Frequency)          how often the word appears IN this document
    IDF (Inverse Document Freq.)  how RARE the word is across ALL documents

    TF-IDF = TF x IDF

    * a word frequent in ONE document but rare elsewhere -> HIGH score
      (it is typical for that document, e.g. "daisies")
    * a word that appears in EVERY document              -> LOW score
      (it does not help to tell documents apart, e.g. "her")

scikit-learn formula (default settings, so you can check a value by hand):
    idf(word) = ln( (1 + n_documents) / (1 + n_documents_containing_word) ) + 1
    then every row is normalised so that its length (L2 norm) = 1.

HOW TO RUN
----------
    python ex6_tfidf.py

OUTPUT
------
    * the TF-IDF matrix and the top 5 words per document in the terminal
    * outputs/ex6_tfidf_matrix.csv
    * outputs/ex6_top5_per_document.png
    * outputs/ex6_bow_vs_tfidf.png    (bonus comparison with Bag of Words)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
DEFAULT_FILES = ["Ch1_Ex5-6_doc1.txt", "Ch1_Ex5-6_doc2.txt", "Ch1_Ex5-6_doc3.txt"]

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.precision", 3)


def read_documents(file_names):
    """Step 1 - read every file and keep ALL of them in ONE list.

    (A classic bug: creating `docs = []` INSIDE the loop - then only the last
    file is kept.  The list must be created once, before the loop.)
    """
    names, texts = [], []
    for name in file_names:
        path = Path(name) if Path(name).is_absolute() else HERE / name
        if not path.exists():
            sys.exit(f"File not found: {path}")
        texts.append(path.read_text(encoding="utf-8"))
        names.append(path.stem.replace("Ch1_Ex5-6_", ""))
    return names, texts


def main():
    files = sys.argv[1:] or DEFAULT_FILES
    names, texts = read_documents(files)
    print(f"Read {len(texts)} documents: {names}")

    # Step 2 - TF-IDF matrix ------------------------------------------------
    # We remove English stopwords.  Why?  IDF can only push a word down if
    # there are MANY documents.  With only 3 documents the IDF of a word goes
    # from 1.00 (in every document) to 1.69 (in one document): not enough to
    # stop "the" or "her" from winning in a short text.  (The bonus below keeps
    # the stopwords to show exactly this effect.)
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(texts)

    # Step 3 - DataFrame --------------------------------------------------
    df = pd.DataFrame(tfidf.toarray(), index=names, columns=vectorizer.get_feature_names_out())
    print("\n" + "=" * 80)
    print(f"STEPS 2-3 - TF-IDF matrix: {df.shape[0]} documents x {df.shape[1]} words (values 0 -> 1)")
    print("=" * 80)
    print(df.round(3).to_string())
    df.to_csv(OUT / "ex6_tfidf_matrix.csv", float_format="%.4f")

    # The IDF learned for each word: low = common everywhere, high = rare
    idf = pd.Series(vectorizer.idf_, index=vectorizer.get_feature_names_out()).sort_values()
    print("\nIDF of a few words (low = appears in many documents, high = in one only):")
    print(pd.concat([idf.head(4), idf.tail(4)]).round(3).to_string())

    # Step 4 - top 5 words per document -------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4 - Top 5 words per document (highest TF-IDF)")
    print("=" * 80)
    top5 = {}
    for doc in names:
        top5[doc] = df.loc[doc].sort_values(ascending=False).head(5)
        words = ", ".join(f"{w} ({v:.3f})" for w, v in top5[doc].items())
        print(f"  {doc}: {words}")
    print("  (Equal scores = words that appear once in the same document only: their order")
    print("   in the top 5 is then just alphabetical, not more important.)")

    # Bonus - comparison with Bag of Words ----------------------------------
    # For a fair comparison we use the SAME vocabulary (stopwords included)
    # for both methods.
    full_tfidf = TfidfVectorizer()
    df_full = pd.DataFrame(full_tfidf.fit_transform(texts).toarray(),
                           index=names, columns=full_tfidf.get_feature_names_out())
    bow = pd.DataFrame(CountVectorizer().fit_transform(texts).toarray(),
                       index=names, columns=full_tfidf.get_feature_names_out())
    compare = pd.DataFrame({
        "bow_total_count": bow.sum(axis=0),
        "docs_containing": (bow > 0).sum(axis=0),
        "mean_tfidf": df_full.mean(axis=0),
    })
    print("\n" + "=" * 80)
    print("BONUS - Bag of Words vs TF-IDF")
    print("=" * 80)
    high_bow = compare.sort_values("bow_total_count", ascending=False).head(8)
    print("\nHIGH frequency in BoW ... and what TF-IDF thinks of them:")
    print(high_bow.round(3).to_string())
    everywhere = compare[compare["docs_containing"] == len(names)].index.tolist()
    print(f"\nWords present in ALL {len(names)} documents (lowest IDF): {everywhere}")
    print("-> these words have a HIGH count but a LOW TF-IDF weight in each document.")
    unique = compare[(compare["docs_containing"] == 1) & (compare["bow_total_count"] >= 2)]
    print(f"\nWords that are frequent in ONE document only (highlighted by TF-IDF): "
          f"{unique.index.tolist()}")
    print("-> TF-IDF ranks them first: they are 'unique' to that document.")
    print("\nTop 5 per document if we KEEP the stopwords (to compare with step 4):")
    for doc in names:
        print(f"  {doc}: {', '.join(df_full.loc[doc].sort_values(ascending=False).head(5).index)}")
    print("-> with only 3 documents, 'the', 'her', 'of' still sneak in: IDF needs a large")
    print("   corpus to be really effective, so in practice we often combine TF-IDF + stopwords.")

    # ------------------------------------------------------------------------
    # GRAPH 1: top 5 per document (small multiples, one panel per document)
    # ------------------------------------------------------------------------
    BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#52514e"
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.2), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, doc in zip(axes, names):
        s = top5[doc]
        ax.barh(s.index, s.values, color=BLUE, height=0.6)
        for y, v in enumerate(s.values):
            ax.text(v + 0.005, y, f"{v:.2f}", va="center", color=GREY, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(doc, loc="left", weight="bold")
        ax.set_xlabel("TF-IDF weight")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e5e4e0", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Chapter 1 - Exercise 6: top 5 TF-IDF words per document", weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "ex6_top5_per_document.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------------
    # GRAPH 2: share of each word in BoW vs in TF-IDF (10 most frequent words)
    # "share" = the word's part of the total, in % -> makes both methods comparable.
    # ------------------------------------------------------------------------
    share = pd.DataFrame({
        "bow": bow.sum(axis=0) / bow.values.sum() * 100,
        "tfidf": df_full.sum(axis=0) / df_full.values.sum() * 100,
        "in_all": compare["docs_containing"] == len(names),
    }).loc[compare.sort_values("bow_total_count", ascending=False).index[:10]]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    y = np.arange(len(share))
    h = 0.38
    ax.barh(y - h / 2, share["bow"], h, color=GREY, label="Bag of Words share")
    ax.barh(y + h / 2, share["tfidf"], h, color=BLUE, label="TF-IDF share")
    for yi, (w, r) in zip(y, share.iterrows()):
        change = (r["tfidf"] / r["bow"] - 1) * 100
        tag = "in all 3 docs" if r["in_all"] else "in 1-2 docs"
        ax.text(max(r["bow"], r["tfidf"]) + 0.08, yi, f"{change:+.0f} %   ({tag})", va="center",
                fontsize=9, color=ORANGE if r["in_all"] else GREY)
    ax.set_yticks(y, share.index)
    ax.invert_yaxis()
    ax.set_xlim(0, share[["bow", "tfidf"]].values.max() * 1.45)
    ax.set_xlabel("Share of the word in the whole matrix (%)")
    ax.set_title("Bonus: TF-IDF shrinks the words that appear in EVERY document the most",
                 loc="left", weight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "ex6_bow_vs_tfidf.png", dpi=150)
    plt.close(fig)

    print(f"\nFiles saved in: {OUT}")


if __name__ == "__main__":
    main()
