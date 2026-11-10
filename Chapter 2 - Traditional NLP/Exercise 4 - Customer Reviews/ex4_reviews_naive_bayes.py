"""
Chapter 2 - Exercise 4 : Customer reviews - important / neutral / not important
================================================================================

SAME PIPELINE AS EXERCISE 3, BUT WITH 3 CLASSES
-----------------------------------------------
    text --CountVectorizer--> numbers --MultinomialNB--> important | neutral | not important

A company receives many reviews and wants to answer the IMPORTANT ones first
(broken product, double payment, cannot log in...).  Naive Bayes works the same
with 3 classes as with 2: it computes a probability for every class and keeps
the highest one.

WHAT CHANGES WITH 3 CLASSES?
----------------------------
* precision / recall / F1 are computed PER CLASS.  To get one number we take
  the MACRO average = the simple mean of the 3 classes.  Each class counts the
  same, even the small one ("important" has only 10 reviews).
* The classes are UNBALANCED (21 / 15 / 10).  A lazy model that always says
  "not important" would already get 21/46 = 46 % accuracy -> always compare
  with this "baseline"!
* STOPWORDS ARE NOT ALWAYS USELESS: "but", "not", "no" are in the English
  stopword list, yet they are what makes a review "neutral" ("nice, BUT slow").
  The script trains a third model without stopwords to show the damage.
* The confusion matrix is 3 x 3.  For this business the worst mistake is an
  "important" review predicted as "not important" (an angry customer waits).

HOW TO RUN
----------
    python ex4_reviews_naive_bayes.py

OUTPUT (outputs/)
-----------------
    ex4_confusion_matrices.png
    ex4_count_vs_tfidf.png
    ex4_top_words_per_class.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

DATA_FILE = HERE / "Ch2_Ex4_Reviews.csv"
CLASSES = ["important", "neutral", "not important"]
RANDOM_STATE = 42
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"


def run(name, make_vectorizer, X_train, X_test, y_train, y_test, X_all, y_all):
    pipe = make_pipeline(make_vectorizer(), MultinomialNB(alpha=1.0))
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Cross-validation: every review is predicted ONCE by a model that did not
    # see it during training -> a confusion matrix over all 46 reviews.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_cv = cross_val_predict(make_pipeline(make_vectorizer(), MultinomialNB()), X_all, y_all, cv=cv)

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print("--- single train/test split ---")
    print(classification_report(y_test, y_pred, labels=CLASSES, zero_division=0))
    print("--- 5-fold cross-validation (all 46 reviews) ---")
    print(classification_report(y_all, y_cv, labels=CLASSES, zero_division=0))

    return {"name": name, "pipe": pipe, "y_pred": y_pred, "y_cv": y_cv,
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "cv_accuracy": accuracy_score(y_all, y_cv),
            "cv_macro_f1": f1_score(y_all, y_cv, average="macro", zero_division=0)}


def main():
    # 1. Load -------------------------------------------------------------------
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} reviews from {DATA_FILE.name}")
    counts = df["label"].value_counts()
    print(counts.to_string())
    baseline = counts.max() / len(df)
    print(f"\nBaseline (always predict '{counts.idxmax()}'): {baseline:.1%} accuracy")
    for lab in CLASSES:
        print(f"  [{lab}] {df.loc[df['label'] == lab, 'text'].iloc[0]}")

    # 3. Split ---------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=RANDOM_STATE, stratify=df["label"])
    print(f"\nTrain: {len(X_train)} reviews   Test: {len(X_test)} reviews")

    # 2 + 4 + 5 -------------------------------------------------------------------
    # IMPORTANT: we keep the stopwords.  The English stopword list contains
    # "but", "not", "no" - exactly the words that make a review NEUTRAL
    # ("good BUT could be faster").  The third model proves it.
    results = [
        run("CountVectorizer + Naive Bayes", lambda: CountVectorizer(),
            X_train, X_test, y_train, y_test, df["text"], df["label"]),
        run("TfidfVectorizer + Naive Bayes (challenge)", lambda: TfidfVectorizer(),
            X_train, X_test, y_train, y_test, df["text"], df["label"]),
        run("CountVectorizer without stopwords + Naive Bayes (extra)",
            lambda: CountVectorizer(stop_words="english"),
            X_train, X_test, y_train, y_test, df["text"], df["label"]),
    ]

    table = pd.DataFrame(results).set_index("name")[
        ["test_accuracy", "test_macro_f1", "cv_accuracy", "cv_macro_f1"]]
    print("\n" + "=" * 70)
    print(f"CHALLENGE - Count vs TF-IDF   (baseline accuracy {baseline:.1%})")
    print("=" * 70)
    print((table * 100).round(1).astype(str).add(" %").to_string())
    count_r, tfidf_r, nostop_r = results
    print("\nInterpretation:")
    print(f"  * Count: {count_r['cv_accuracy']:.0%}, TF-IDF: {tfidf_r['cv_accuracy']:.0%} cross-validated accuracy,"
          f" vs {baseline:.0%} for the lazy baseline.")
    print(f"  * Removing stopwords: {nostop_r['cv_accuracy']:.0%} -> "
          + ("WORSE than the baseline!  'but', 'not', 'no' are stopwords, and they are the"
             if nostop_r["cv_accuracy"] < baseline else "worse: 'but', 'not', 'no' are stopwords, and they are the"))
    print("    clues for 'neutral' ('good BUT ...').  Lesson: never remove stopwords blindly,")
    print("    always TEST whether it helps.")
    if tfidf_r["cv_accuracy"] <= count_r["cv_accuracy"]:
        print("  * TF-IDF does not help here: the reviews are one sentence long, so 'how often'")
        print("    a word appears is almost always 1 - there is little to re-weight.")
    else:
        print("  * TF-IDF helps a little: it lowers the weight of words found in many reviews.")
    print("  * 46 reviews is very little: a difference of 1 review = 2 %.  Every review uses")
    print("    almost unique words, so the model has to guess on many unseen words.")

    new = ["I was charged three times and nobody answers my emails!",
           "The package came on Tuesday.",
           "Love the colour, thanks!"]
    best = max(results, key=lambda r: r["cv_macro_f1"])
    print(f"\nPredictions of '{best['name']}' on new reviews:")
    for t, p in zip(new, best["pipe"].predict(new)):
        print(f"  {p:<14} <- {t}")

    # ------------------------------------------------------------------------
    # GRAPHS
    # ------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, r in zip(axes, results):
        cm = confusion_matrix(df["label"], r["y_cv"], labels=CLASSES)
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        ax.set_xticks(range(3), [f"pred.\n{c}" for c in CLASSES])
        ax.set_yticks(range(3), [f"true {c}" for c in CLASSES])
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > cm.max() / 2 else "#0b0b0b")
        ax.set_title(f"{r['name'].split(' +')[0]}\ncross-val. accuracy {r['cv_accuracy']:.0%}",
                     loc="left", weight="bold", fontsize=10)
    fig.suptitle("Chapter 2 - Exercise 4: confusion matrices (5-fold cross-validation, all 46 reviews)",
                 weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "ex4_confusion_matrices.png", dpi=150)
    plt.close(fig)

    metrics = ["test_accuracy", "test_macro_f1", "cv_accuracy", "cv_macro_f1"]
    nice = ["Test split\naccuracy", "Test split\nmacro F1", "Cross-val.\naccuracy", "Cross-val.\nmacro F1"]
    x = np.arange(len(metrics))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 5))
    for k, (r, color) in enumerate(zip(results, [BLUE, ORANGE, AQUA])):
        bars = ax.bar(x + (k - 1) * w, [r[m] * 100 for m in metrics], w, color=color,
                      label=r["name"].split(" +")[0])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.0f}",
                    ha="center", fontsize=8, color=GREY)
    ax.axhline(baseline * 100, color=GREY, linestyle="--", linewidth=1)
    ax.text(-0.45, baseline * 100 + 1.5, f"baseline {baseline:.0%}", ha="left",
            fontsize=9, color=GREY)
    ax.set_xticks(x, nice)
    ax.set_ylim(0, 125)
    ax.set_ylabel("Score (%)")
    ax.set_title("Count vs TF-IDF vs stopwords removed (dashed line = always guess the biggest class)",
                 loc="left", weight="bold")
    ax.legend(frameon=False, loc="upper center", ncol=3)
    ax.set_yticks(range(0, 101, 20))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "ex4_count_vs_tfidf.png", dpi=150)
    plt.close(fig)

    # Most typical words per class (highest probability vs the other classes)
    vec, nb = results[0]["pipe"][0], results[0]["pipe"][1]
    words = vec.get_feature_names_out()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, cls, color in zip(axes, CLASSES, [ORANGE, GREY, BLUE]):
        i = list(nb.classes_).index(cls)
        others = np.delete(nb.feature_log_prob_, i, axis=0).max(axis=0)
        score = nb.feature_log_prob_[i] - others
        top = np.argsort(score)[-8:][::-1]
        ax.barh(words[top], score[top], color=color, height=0.6)
        ax.invert_yaxis()
        ax.set_title(f"Typical words: {cls}", loc="left", weight="bold")
        ax.set_xlabel("log ratio vs the other classes")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e5e4e0", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("What did Naive Bayes learn? (CountVectorizer model)", weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "ex4_top_words_per_class.png", dpi=150)
    plt.close(fig)

    print(f"\nGraphs saved in: {OUT}")


if __name__ == "__main__":
    main()
