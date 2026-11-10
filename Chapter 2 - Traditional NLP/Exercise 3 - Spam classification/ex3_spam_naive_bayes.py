"""
Chapter 2 - Exercise 3 : Spam classification with Multinomial Naive Bayes
==========================================================================

FROM RULES TO MACHINE LEARNING
------------------------------
VADER (Exercises 1-2) uses rules written by humans.  Here the computer LEARNS
the rules itself from examples that are already labelled ("spam" / "ham").
ham = a normal, wanted e-mail.

THE PIPELINE (the same for almost every text classifier)
--------------------------------------------------------
    text  --CountVectorizer-->  numbers  --train-->  model  --predict-->  label

    1. LOAD      the CSV (columns: text, label)
    2. VECTORIZE CountVectorizer = Bag of Words (Chapter 1 - Exercise 5)
    3. SPLIT     train_test_split: the model learns on the TRAINING set and is
                 evaluated on the TEST set that it has NEVER seen.
                 (Testing on the training data is like giving students the
                 exam questions before the exam: the score means nothing.)
    4. TRAIN     MultinomialNB().fit(X_train, y_train)
    5. MEASURE   accuracy, precision, recall, F1, confusion matrix

HOW DOES NAIVE BAYES DECIDE?
----------------------------
During training it counts, for each class, how often every word appears.
    P("free" | spam) is high,   P("free" | ham) is low
For a new e-mail it multiplies the probabilities of all its words for each
class and chooses the class with the highest result.  "Naive" = it assumes
that the words are independent of each other (not true, but it works well!).
alpha=1 ("Laplace smoothing") avoids a probability of 0 for an unseen word.

THE METRICS
-----------
    accuracy  = correct predictions / all predictions
    precision = of the e-mails predicted SPAM, how many really were spam?
                (low precision = good e-mails end up in the spam folder!)
    recall    = of all the real spam, how many did we catch?
    F1        = one number that combines precision and recall

WHY CROSS-VALIDATION?
---------------------
The dataset only has 48 e-mails, so a test set has ~12 e-mails: one lucky or
unlucky split changes the accuracy a lot.  5-fold cross-validation splits the
data 5 times, tests on every part once and gives the AVERAGE -> more reliable.

HOW TO RUN
----------
    python ex3_spam_naive_bayes.py

OUTPUT (outputs/)
-----------------
    ex3_confusion_matrices.png   Count vs TF-IDF vs "stopwords removed"
    ex3_count_vs_tfidf.png       comparison of the metrics (challenge)
    ex3_top_words.png            the words that "scream" spam or ham
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import clone
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

DATA_FILE = HERE / "Ch2_Ex3_Spam.csv"
POSITIVE = "spam"          # the class we want to detect (used for precision / recall)
RANDOM_STATE = 42          # fixed seed -> same split every time you run the script
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"


def evaluate(name, vectorizer, X_train, X_test, y_train, y_test, X_all, y_all):
    """Train Naive Bayes with the given vectorizer and return all the metrics."""
    # A PIPELINE glues the vectorizer and the model together:
    #   pipe.fit(texts)      -> vectorizer.fit_transform + model.fit
    #   pipe.predict(texts)  -> vectorizer.transform     + model.predict
    # This guarantees the vocabulary is learned on the TRAINING texts only.
    pipe = make_pipeline(clone(vectorizer), MultinomialNB(alpha=1.0))
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    # clone() = a fresh, untrained copy of the vectorizer with the same settings
    cv_scores = cross_val_score(make_pipeline(clone(vectorizer), MultinomialNB(alpha=1.0)),
                                X_all, y_all, cv=cv, scoring="accuracy")

    print("\n" + "=" * 70)
    print(f"{name}")
    print("=" * 70)
    print(f"Vocabulary size: {len(pipe[0].vocabulary_)} words")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"5-fold cross-validation accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})"
          f"   folds: {[f'{s:.0%}' for s in cv_scores]}")

    return {
        "name": name,
        "pipe": pipe,
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=POSITIVE, zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label=POSITIVE, zero_division=0),
        "f1": f1_score(y_test, y_pred, pos_label=POSITIVE, zero_division=0),
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
    }


def main():
    # 1. LOAD -----------------------------------------------------------------
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} e-mails from {DATA_FILE.name}")
    print(df["label"].value_counts().to_string())
    print("\nExamples:")
    for lab in df["label"].unique():
        print(f"  [{lab}] {df.loc[df['label'] == lab, 'text'].iloc[0]}")

    # 3. SPLIT (stratify = keep the same spam/ham proportion in train and test) --
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=RANDOM_STATE, stratify=df["label"])
    print(f"\nTrain: {len(X_train)} e-mails   Test: {len(X_test)} e-mails")

    # 2 + 4 + 5. VECTORIZE, TRAIN, MEASURE -------------------------------------
    # We KEEP the stopwords: in short messages words like "you", "your", "now"
    # are typical of spam.  The third model shows what happens if we remove them.
    results = [
        evaluate("CountVectorizer + Naive Bayes", CountVectorizer(),
                 X_train, X_test, y_train, y_test, df["text"], df["label"]),
        # Challenge: TF-IDF instead of counts
        evaluate("TfidfVectorizer + Naive Bayes (challenge)", TfidfVectorizer(),
                 X_train, X_test, y_train, y_test, df["text"], df["label"]),
        # Extra lesson: remove the English stopwords
        evaluate("CountVectorizer without stopwords + Naive Bayes (extra)",
                 CountVectorizer(stop_words="english"),
                 X_train, X_test, y_train, y_test, df["text"], df["label"]),
    ]

    # Comparison table --------------------------------------------------------
    table = pd.DataFrame(results).set_index("name")[["accuracy", "precision", "recall", "f1", "cv_mean"]]
    print("\n" + "=" * 70)
    print("CHALLENGE - CountVectorizer vs TfidfVectorizer")
    print("=" * 70)
    print((table * 100).round(1).astype(str).add(" %").to_string())
    count_cv, tfidf_cv, nostop_cv = (r["cv_mean"] for r in results)
    print("\nInterpretation:")
    if abs(count_cv - tfidf_cv) < 0.03:
        print("  * Count and TF-IDF are practically equal here: the spam words ('win',")
        print("    'free', 'claim', 'urgent') are obvious for both representations.")
    else:
        better = "TF-IDF" if tfidf_cv > count_cv else "Count"
        print(f"  * {better} is better here ({max(count_cv, tfidf_cv):.0%} vs {min(count_cv, tfidf_cv):.0%}).")
    print("  * TF-IDF lowers the weight of words that appear everywhere; it helps more")
    print("    on large datasets with long texts.")
    print(f"  * Removing stopwords: {nostop_cv:.0%} instead of {count_cv:.0%} -> "
          + ("WORSE: in spam, small words like 'you', 'your', 'now' are useful clues!"
             if nostop_cv < count_cv else "no loss here."))
    print("  * Trust the cross-validation column more than the single test split (12 e-mails!).")
    best_test = max(results, key=lambda r: r["accuracy"])["name"]
    best_cv = max(results, key=lambda r: r["cv_mean"])["name"]
    if best_test != best_cv:
        print(f"    Look: the single split says the best model is '{best_test.split(' +')[0]}',")
        print(f"    cross-validation says '{best_cv.split(' +')[0]}'.  One split of 12 e-mails can")
        print("    be lucky or unlucky - this is exactly why we use cross-validation.")

    # Try the model on new e-mails ----------------------------------------------
    best = max(results, key=lambda r: r["cv_mean"])
    new_mails = ["You have been selected for a FREE cruise, reply now!",
                 "Can you send me the slides from today's lecture?"]
    print(f"\nPredictions of the '{best['name']}' model on new e-mails:")
    for m, p, proba in zip(new_mails, best["pipe"].predict(new_mails), best["pipe"].predict_proba(new_mails)):
        print(f"  {p:<5} ({proba.max():.0%} sure)  <- {m}")

    # ------------------------------------------------------------------------
    # GRAPHS
    # ------------------------------------------------------------------------
    labels = sorted(df["label"].unique())

    # Confusion matrices side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, r in zip(axes, results):
        cm = confusion_matrix(y_test, r["y_pred"], labels=labels)
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        ax.set_xticks(range(len(labels)), [f"pred. {l}" for l in labels])
        ax.set_yticks(range(len(labels)), [f"true {l}" for l in labels])
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                        color="white" if cm[i, j] > cm.max() / 2 else "#0b0b0b")
        ax.set_title(f"{r['name'].split(' +')[0]}\ntest accuracy {r['accuracy']:.0%}"
                     f" | cross-val. {r['cv_mean']:.0%}", loc="left", weight="bold", fontsize=10)
    fig.suptitle("Chapter 2 - Exercise 3: confusion matrices on the test set (diagonal = correct)",
                 weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "ex3_confusion_matrices.png", dpi=150)
    plt.close(fig)

    # Metric comparison
    metrics = ["accuracy", "precision", "recall", "f1", "cv_mean"]
    nice = ["Accuracy", "Precision\n(spam)", "Recall\n(spam)", "F1\n(spam)", "Cross-val.\naccuracy"]
    x = np.arange(len(metrics))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 5))
    for k, (r, color) in enumerate(zip(results, [BLUE, ORANGE, AQUA])):
        offset = (k - 1) * w
        vals = [r[m] * 100 for m in metrics]
        bars = ax.bar(x + offset, vals, w, color=color, label=r["name"].split(" +")[0])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.0f}",
                    ha="center", fontsize=8, color=GREY)
        ax.errorbar([x[-1] + offset], [r["cv_mean"] * 100], yerr=[r["cv_std"] * 100],
                    fmt="none", ecolor=GREY, capsize=4)
    ax.set_xticks(x, nice)
    ax.set_ylim(0, 125)
    ax.set_ylabel("Score (%)")
    ax.set_title("Count vs TF-IDF vs stopwords removed (error bar = spread over the 5 folds)",
                 loc="left", weight="bold")
    ax.legend(frameon=False, loc="upper center", ncol=3)
    ax.set_yticks(range(0, 101, 20))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "ex3_count_vs_tfidf.png", dpi=150)
    plt.close(fig)

    # Most informative words: log P(word|spam) - log P(word|ham)
    pipe = results[0]["pipe"]
    vec, nb = pipe[0], pipe[1]
    words = vec.get_feature_names_out()
    spam_idx = list(nb.classes_).index(POSITIVE)
    diff = nb.feature_log_prob_[spam_idx] - nb.feature_log_prob_[1 - spam_idx]
    order = np.argsort(diff)
    top_ham, top_spam = order[:10], order[-10:][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, idx, color, ttl in [(axes[0], top_spam, ORANGE, "Words that point to SPAM"),
                                (axes[1], top_ham, BLUE, "Words that point to HAM")]:
        ax.barh(words[idx], np.abs(diff[idx]), color=color, height=0.6)
        ax.invert_yaxis()
        ax.set_xlabel("How much more likely in this class (log ratio)")
        ax.set_title(ttl, loc="left", weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e5e4e0", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("What did Naive Bayes learn? (CountVectorizer model)", weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "ex3_top_words.png", dpi=150)
    plt.close(fig)

    print(f"\nGraphs saved in: {OUT}")


if __name__ == "__main__":
    main()
