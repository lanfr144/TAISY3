"""
Chapter 2 - Exercise 5 : Topic Analysis with NMF on the 20 Newsgroups dataset
==============================================================================

KEY IDEAS FOR A JUNIOR
----------------------
TOPIC MODELLING = finding the THEMES of a big collection of texts WITHOUT
telling the computer what the themes are (= UNSUPERVISED learning).
Exercises 3-4 were SUPERVISED: we gave the right labels.  Here we only give
the texts, and the algorithm discovers groups of words that go together.

NMF (Non-negative Matrix Factorization), in one picture:

     TF-IDF matrix          W (documents x topics)    H (topics x words)
     (docs x words)   ~=    "how much of each topic   "which words make
                             is in each document"      each topic"
       18 000 x 5 000   ~=        18 000 x 20       x      20 x 5 000

All numbers are >= 0 ("non-negative"), so we can read them as "how much".
A topic = a row of H.  Its 10 highest words describe the topic, for example
    god, jesus, bible, christian, church ...  -> clearly "religion"
YOU give the topic a name: the algorithm only gives the words.

THE DATASET
-----------
20 Newsgroups = ~18 000 messages from 20 discussion forums (1990s internet),
for example sci.space, rec.sport.hockey, comp.graphics.  It is built into
scikit-learn (downloaded automatically the first time, ~14 MB, then cached).
We remove headers, signatures and quotes, otherwise NMF finds topics such as
"e-mail addresses" instead of real themes.

HOW TO RUN
----------
    python ex5_topic_analysis_nmf.py
    python ex5_topic_analysis_nmf.py --topics 10       (try another number of topics)

OUTPUT (outputs/)
-----------------
    ex5_dataset_overview.png      task 1: how many posts per category
    ex5_top_words_per_topic.png   task 4: the 10 top words of each topic
    ex5_topic_vs_category.png     extra: does a topic match a real category?
    ex5_topics.csv                the top words in a table
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
BLUE, GREY = "#2a78d6", "#52514e"


def load_data():
    """Task 1 - load the dataset (needs internet the very first time)."""
    from sklearn.datasets import fetch_20newsgroups
    try:
        return fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"),
                                  shuffle=True, random_state=42)
    except Exception as err:   # no internet, proxy, ...
        sys.exit(f"Could not download 20 Newsgroups ({err}).\n"
                 "Connect to the internet once; scikit-learn keeps a copy in ~/scikit_learn_data.")


def main():
    parser = argparse.ArgumentParser(description="NMF topic modelling on 20 Newsgroups")
    parser.add_argument("--topics", type=int, default=20, help="number of topics (default 20)")
    parser.add_argument("--words", type=int, default=10, help="top words per topic (default 10)")
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Task 1 - Load and explore
    # ---------------------------------------------------------------------
    data = load_data()
    texts = [t for t in data.data]
    df = pd.DataFrame({"text": texts, "category": [data.target_names[i] for i in data.target]})
    df["n_words"] = df["text"].str.split().str.len()

    print("=" * 80)
    print("TASK 1 - LOAD AND EXPLORE")
    print("=" * 80)
    print(f"Number of posts      : {len(df):,}")
    print(f"Number of categories : {df['category'].nunique()}")
    print(f"Empty posts          : {(df['n_words'] == 0).sum():,}  (only quotes / headers, now removed)")
    print(f"Words per post       : median {df['n_words'].median():.0f}, mean {df['n_words'].mean():.0f}, "
          f"max {df['n_words'].max():,}")
    print("\nPosts per category:")
    print(df["category"].value_counts().to_string())
    example = df[df["n_words"].between(30, 80)].iloc[0]
    print(f"\nExample post from [{example['category']}]:\n  {' '.join(example['text'].split())[:400]}")

    # Drop empty posts: they carry no information and only add zero rows.
    df = df[df["n_words"] > 0].reset_index(drop=True)

    # ---------------------------------------------------------------------
    # Task 2 - Preprocess and vectorise with TF-IDF
    # ---------------------------------------------------------------------
    # Settings explained:
    #   stop_words="english"  remove "the", "and" ... (they would be in EVERY topic)
    #   max_df=0.95           ignore words present in more than 95 % of the posts
    #   min_df=2              ignore words present in only 1 post (typos, names)
    #   max_features=5000     keep the 5000 most frequent words (speed + memory)
    #   token_pattern         words of 3+ LETTERS only (no numbers like "1993")
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.95, min_df=2, max_features=5000,
                                 token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
    X = vectorizer.fit_transform(df["text"])
    words = vectorizer.get_feature_names_out()
    print("\n" + "=" * 80)
    print("TASK 2 - TF-IDF")
    print("=" * 80)
    print(f"TF-IDF matrix: {X.shape[0]:,} posts x {X.shape[1]:,} words")
    print(f"Only {X.nnz / (X.shape[0] * X.shape[1]):.2%} of the cells are not zero (a 'sparse' matrix).")

    # ---------------------------------------------------------------------
    # Task 3 - NMF with as many topics as categories (20)
    # ---------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"TASK 3 - NMF with {args.topics} topics (this can take ~30 s)")
    print("=" * 80)
    # init="nndsvda" = a smart, deterministic start -> same topics every run
    nmf = NMF(n_components=args.topics, init="nndsvda", random_state=42, max_iter=400)
    W = nmf.fit_transform(X)          # posts  x topics
    H = nmf.components_               # topics x words
    print(f"W (posts x topics) = {W.shape},   H (topics x words) = {H.shape}")
    print(f"Reconstruction error: {nmf.reconstruction_err_:.1f}  (lower = the topics explain the data better)")

    # ---------------------------------------------------------------------
    # Task 4 - Top 10 words per topic
    # ---------------------------------------------------------------------
    df["topic"] = W.argmax(axis=1)            # main topic of every post
    topic_rows = []
    print("\n" + "=" * 80)
    print(f"TASK 4 - TOP {args.words} WORDS PER TOPIC  (+ the category that uses it most)")
    print("=" * 80)
    for k in range(args.topics):
        top_idx = H[k].argsort()[::-1][:args.words]
        top_words = [words[i] for i in top_idx]
        members = df[df["topic"] == k]
        main_cat = members["category"].value_counts()
        cat, share = (main_cat.index[0], main_cat.iloc[0] / len(members)) if len(members) else ("-", 0)
        topic_rows.append({"topic": k, "top_words": ", ".join(top_words), "n_posts": len(members),
                           "main_category": cat, "category_share": round(share, 3)})
        print(f"Topic {k:2d} ({len(members):5,} posts, {share:4.0%} {cat:<25}): {' '.join(top_words)}")
    pd.DataFrame(topic_rows).to_csv(OUT / "ex5_topics.csv", index=False)

    print("""
HOW TO INTERPRET
  * Read the 10 words and give the topic a human name (religion, space, hockey...).
  * Topics with a high % for one category = NMF rediscovered that forum by itself,
    without ever seeing the labels!
  * Some forums talk about the same thing (comp.sys.ibm.pc.hardware and
    comp.sys.mac.hardware; talk.religion.misc and soc.religion.christian), so NMF
    often merges them into one topic, and splits another forum in two.
  * Topics full of general words (like, know, think, people) collect the short,
    chatty posts that have no clear theme.""")

    # ---------------------------------------------------------------------
    # GRAPHS
    # ---------------------------------------------------------------------
    # 1. Dataset overview
    counts = df["category"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(counts.index, counts.values, color=BLUE, height=0.65)
    ax.set_xlabel("Number of posts")
    ax.set_title(f"Task 1: 20 Newsgroups - {len(df):,} posts in {len(counts)} categories",
                 loc="left", weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "ex5_dataset_overview.png", dpi=150)
    plt.close(fig)

    # 2. Small multiples: top words per topic
    cols = 5
    rows = int(np.ceil(args.topics / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.1 * rows))
    for k, ax in enumerate(np.atleast_1d(axes).ravel()):
        if k >= args.topics:
            ax.axis("off")
            continue
        top_idx = H[k].argsort()[::-1][:args.words]
        ax.barh([words[i] for i in top_idx], H[k][top_idx], color=BLUE, height=0.7)
        ax.invert_yaxis()
        ax.set_title(f"Topic {k} - {topic_rows[k]['main_category']}", loc="left", fontsize=9, weight="bold")
        ax.tick_params(axis="both", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Task 4: top {args.words} words of each NMF topic "
                 "(title = the category that uses the topic most)", weight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "ex5_top_words_per_topic.png", dpi=130)
    plt.close(fig)

    # 3. Heatmap topic x category (row-normalised: % of each category's posts)
    ct = pd.crosstab(df["category"], df["topic"], normalize="index")
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(ct.values * 100, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(ct.shape[1]), [f"T{c}" for c in ct.columns])
    ax.set_yticks(range(ct.shape[0]), ct.index)
    ax.set_xlabel("NMF topic (main topic of each post)")
    ax.set_title("Which topics does each real category use?  (row = 100 % of the category's posts)",
                 loc="left", weight="bold")
    fig.colorbar(im, ax=ax, label="% of the category's posts")
    fig.tight_layout()
    fig.savefig(OUT / "ex5_topic_vs_category.png", dpi=150)
    plt.close(fig)

    print(f"\nFiles saved in: {OUT}")


if __name__ == "__main__":
    main()
