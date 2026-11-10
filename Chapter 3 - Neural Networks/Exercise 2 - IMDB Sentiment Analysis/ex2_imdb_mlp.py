"""
Chapter 3 - Exercise 2 : IMDB sentiment analysis with Neural Networks (MLP)
============================================================================

THE STORY OF THE IMDB DATASET IN THIS COURSE
--------------------------------------------
    Chapter 2 - Ex. 2   VADER (rules, no training)        -> roughly 70 % accuracy
    Chapter 3 - Ex. 2   MLP (learns from 1000s of examples) -> let's see!

THE PIPELINE
------------
    1. load the CSV                         (review, sentiment)
    2. remove the HTML tags                 "<br /><br />"
    3. TF-IDF  -> document-term matrix      (Chapter 1 - Exercise 6)
    4. MLP     -> positive / negative (/ neutral, see below)
    5. try DIFFERENT PARAMETERS for the MLP and compare

HYPER-PARAMETERS = the settings YOU choose before training
----------------------------------------------------------
    hidden_layer_sizes  (50,) = 1 layer of 50 neurons, (100, 50) = 2 layers
                        more neurons = more capacity, but slower and can overfit
    activation          "relu" (default) or "logistic" (sigmoid)
    alpha               L2 regularisation: a penalty on big weights.  Higher alpha
                        = simpler model = less overfitting
    early_stopping=True keeps 10 % of the training data aside and STOPS when the
                        score on it does not improve any more -> avoids overfitting
                        and saves time

OVERFITTING - how to see it
---------------------------
    train accuracy 100 %  but  test accuracy 85 %  -> the network learned the
    training reviews by heart instead of learning "what makes a review positive".
    The script prints both so you can compare.

"NEUTRAL"?
----------
IMDB only contains positive and negative reviews, so the network learns two
classes.  But it gives a PROBABILITY: P(positive) = 0.52 means "not sure".
We call a review neutral when 0.4 < P(positive) < 0.6 and check that those
"unsure" reviews are indeed the ones the network gets wrong most often.

HOW TO RUN
----------
    python ex2_imdb_mlp.py                  (all 50 000 reviews, about 1-3 minutes)
    python ex2_imdb_mlp.py --sample 10000   (quick test, ~30 s - but MORE overfitting:
                                             compare the "gap" column of both runs!)

OUTPUT (outputs/)
-----------------
    ex2_mlp_parameters.png        test vs train accuracy for each setting
    ex2_loss_curves.png           how fast each network learns
    ex2_confusion_matrix.png      best network, with the neutral band
    ex2_probability_histogram.png how sure is the network?
    ex2_results.csv
"""

import argparse
import html
import re
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_CANDIDATES = [ROOT / "data" / "Ch2_Ex1_IMDB.csv", ROOT / "data" / "Ch2_Ex1_IMDB.csv.zip"]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GREY = "#52514e"
COLORS = {"positive": "#1baf7a", "neutral": "#9a9890", "negative": "#e34948"}

# The MLP settings we compare (task 5).  Feel free to add your own!
MLP_CONFIGS = {
    "50 neurons, relu":            dict(hidden_layer_sizes=(50,), activation="relu", alpha=1e-4),
    "100 neurons, relu":           dict(hidden_layer_sizes=(100,), activation="relu", alpha=1e-4),
    "100 neurons, logistic":       dict(hidden_layer_sizes=(100,), activation="logistic", alpha=1e-4),
    "100+50 neurons (2 layers)":   dict(hidden_layer_sizes=(100, 50), activation="relu", alpha=1e-4),
    "50 neurons, strong alpha":    dict(hidden_layer_sizes=(50,), activation="relu", alpha=1e-1),
}


def load_imdb(sample):
    for path in DATA_CANDIDATES:
        if path.exists():
            df = pd.read_csv(path)
            print(f"Loaded {len(df):,} reviews from {path.relative_to(ROOT)}")
            break
    else:
        sys.exit("IMDB dataset not found.  Put Ch2_Ex1_IMDB.csv (or .csv.zip) in EXERCISES/data/")
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"Using a random sample of {len(df):,} reviews (run without --sample for all)")
    return df


TAG = re.compile(r"<[^>]+>")


def clean_html(text):
    """Task 2 - remove HTML tags and entities."""
    return re.sub(r"\s+", " ", html.unescape(TAG.sub(" ", text))).strip()


def to_label(p_positive, band=0.1):
    """P(positive) -> label, with a 'neutral' zone 0.5 +/- band."""
    return np.where(p_positive >= 0.5 + band, "positive",
                    np.where(p_positive <= 0.5 - band, "negative", "neutral"))


def main():
    parser = argparse.ArgumentParser(description="MLP sentiment analysis on IMDB")
    parser.add_argument("--sample", type=int, default=0, help="number of reviews (0 = all 50 000)")
    parser.add_argument("--features", type=int, default=5_000, help="TF-IDF vocabulary size")
    args = parser.parse_args()

    # 1 + 2. Load and clean -----------------------------------------------------
    df = load_imdb(args.sample or None)
    df["clean"] = df["review"].astype(str).apply(clean_html)
    print(f"Task 2: HTML removed.  Balance: {df['sentiment'].value_counts().to_dict()}")

    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        df["clean"], df["sentiment"], test_size=0.2, random_state=RANDOM_STATE, stratify=df["sentiment"])

    # 3. TF-IDF document-term matrix ------------------------------------------------
    #   ngram_range=(1, 2): single words AND pairs of words ("not good", "must see")
    #   sublinear_tf=True : 1 + log(count) -> a word repeated 20 times does not dominate
    vectorizer = TfidfVectorizer(max_features=args.features, ngram_range=(1, 2), min_df=2,
                                 stop_words=None, sublinear_tf=True)
    X_train = vectorizer.fit_transform(X_train_txt)   # learn the vocabulary on TRAIN only
    X_test = vectorizer.transform(X_test_txt)
    print(f"Task 3: TF-IDF matrix train {X_train.shape}, test {X_test.shape}")
    print("        (we keep stopwords: 'not', 'no', 'but' are vital for sentiment, see Ch.2 Ex.4)")

    # Reference model: Naive Bayes (Chapter 2) ---------------------------------
    nb = MultinomialNB().fit(X_train, y_train)
    nb_acc = accuracy_score(y_test, nb.predict(X_test))
    print(f"\nReference - Naive Bayes on the same TF-IDF: {nb_acc:.2%} test accuracy")

    # 4 + 5. MLPs with different parameters ------------------------------------------
    print("\n" + "=" * 90)
    print("TASKS 4-5 - MLP WITH DIFFERENT PARAMETERS")
    print("=" * 90)
    print(f"{'setting':<28}{'train acc':>10}{'test acc':>10}{'gap':>7}{'epochs':>8}{'time':>8}")
    rows, models = [], {}
    for name, params in MLP_CONFIGS.items():
        mlp = MLPClassifier(**params, solver="adam", max_iter=50, early_stopping=True,
                            validation_fraction=0.1, n_iter_no_change=3, random_state=RANDOM_STATE)
        t0 = time.time()
        mlp.fit(X_train, y_train)
        seconds = time.time() - t0
        train_acc = accuracy_score(y_train, mlp.predict(X_train))
        test_acc = accuracy_score(y_test, mlp.predict(X_test))
        rows.append({"setting": name, "train_accuracy": train_acc, "test_accuracy": test_acc,
                     "overfit_gap": train_acc - test_acc, "epochs": mlp.n_iter_, "seconds": seconds})
        models[name] = mlp
        print(f"{name:<28}{train_acc:>10.2%}{test_acc:>10.2%}{train_acc - test_acc:>7.1%}"
              f"{mlp.n_iter_:>8}{seconds:>7.0f}s")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "ex2_results.csv", index=False)

    best = res.loc[res["test_accuracy"].idxmax()]
    worst_gap = res.loc[res["overfit_gap"].idxmax()]
    print(f"\nBest setting: {best['setting']} -> {best['test_accuracy']:.2%} "
          f"(Naive Bayes: {nb_acc:.2%})")
    print(f"Biggest overfitting gap: {worst_gap['setting']} "
          f"(train {worst_gap['train_accuracy']:.1%} vs test {worst_gap['test_accuracy']:.1%})")
    spread = res["test_accuracy"].max() - res["test_accuracy"].min()
    if spread < 0.02:
        print(f"All settings are within {spread:.1%} of each other: the DATA (TF-IDF features, number")
        print("of reviews) matters more than the exact network size.")
    biggest = "100+50 neurons (2 layers)"
    if biggest in set(res["setting"]) and best["setting"] != biggest:
        print("The biggest network is NOT the best one: more neurons = more capacity to overfit.")
    print(f"Average overfitting gap: {res['overfit_gap'].mean():.1%} "
          "(try --sample 10000: with less data the gap grows).")

    # Neutral band with the best network ------------------------------------------------
    mlp = models[best["setting"]]
    pos_col = list(mlp.classes_).index("positive")
    p_pos = mlp.predict_proba(X_test)[:, pos_col]
    pred3 = to_label(p_pos, band=0.1)
    y_true = y_test.to_numpy()
    sure = pred3 != "neutral"
    print("\n" + "=" * 90)
    print("POSITIVE / NEGATIVE / NEUTRAL with the best network (neutral = 0.4 < P(positive) < 0.6)")
    print("=" * 90)
    print(f"  neutral (unsure) reviews         : {(~sure).mean():.1%}")
    print(f"  accuracy on the 'sure' reviews   : {(pred3[sure] == y_true[sure]).mean():.2%}")
    two_class = np.where(p_pos >= 0.5, "positive", "negative")
    print(f"  accuracy of the unsure reviews if we forced a choice: "
          f"{(two_class[~sure] == y_true[~sure]).mean():.1%}  <- close to a coin flip")
    cm = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(pred3, name="predicted")).reindex(
        index=["positive", "negative"], columns=["positive", "neutral", "negative"], fill_value=0)
    print(cm.to_string())

    unsure_idx = np.argsort(np.abs(p_pos - 0.5))[:3]
    print("\nThree reviews the network is LEAST sure about:")
    for i in unsure_idx:
        print(f"  P(pos)={p_pos[i]:.2f} true={y_true[i]:<8} {X_test_txt.iloc[i][:140]}...")

    # ------------------------------------------------------------------------
    # GRAPHS
    # ------------------------------------------------------------------------
    # 1. Parameters: train vs test accuracy (dot plot - the gap = overfitting)
    fig, ax = plt.subplots(figsize=(11, 5))
    y = np.arange(len(res))
    ax.hlines(y, res["test_accuracy"] * 100, res["train_accuracy"] * 100, color="#c3c2b7", linewidth=2)
    ax.scatter(res["train_accuracy"] * 100, y, s=70, color=PALETTE[1], label="train accuracy", zorder=3,
               edgecolor="white", linewidth=1.5)
    ax.scatter(res["test_accuracy"] * 100, y, s=70, color=PALETTE[0], label="test accuracy", zorder=3,
               edgecolor="white", linewidth=1.5)
    for yi, (_, r) in zip(y, res.iterrows()):
        ax.text(r["test_accuracy"] * 100 - 0.4, yi, f"{r['test_accuracy']:.1%}", ha="right", va="center",
                fontsize=9, color=GREY)
    ax.axvline(nb_acc * 100, color=GREY, linestyle="--", linewidth=1)
    ax.text(nb_acc * 100, -0.75, f" Naive Bayes (Ch.2): {nb_acc:.1%}", fontsize=9, color=GREY, va="center")
    ax.set_yticks(y, res["setting"])
    ax.set_ylim(len(res) - 0.5, -1.2)          # inverted axis + room for the label on top
    ax.set_xlabel("Accuracy (%)   - the grey line between the dots = overfitting gap")
    ax.set_title("Task 5: which MLP setting works best?", loc="left", weight="bold")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.0))
    lo = min(nb_acc, res["test_accuracy"].min()) * 100 - 2.5
    ax.set_xlim(lo, res["train_accuracy"].max() * 100 + 0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "ex2_mlp_parameters.png", dpi=150)
    plt.close(fig)

    # 2. Loss curves
    fig, ax = plt.subplots(figsize=(9, 5))
    for color, (name, m) in zip(PALETTE, models.items()):
        ax.plot(range(1, len(m.loss_curve_) + 1), m.loss_curve_, marker="o", markersize=4,
                linewidth=2, color=color, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("Training loss per epoch (early stopping ends each curve)", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "ex2_loss_curves.png", dpi=150)
    plt.close(fig)

    # 3. Confusion matrix with neutral band
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(3), [f"pred. {c}" for c in cm.columns])
    ax.set_yticks(range(2), [f"true {r}" for r in cm.index])
    for i in range(2):
        for j in range(3):
            v = cm.values[i, j]
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=12,
                    color="white" if v > cm.values.max() / 2 else "#0b0b0b")
    ax.set_title(f"Best MLP ({best['setting']}) with a neutral band", loc="left", weight="bold")
    fig.colorbar(im, ax=ax, label="reviews")
    fig.tight_layout()
    fig.savefig(OUT / "ex2_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # 4. Probability histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 41)
    for lab in ["positive", "negative"]:
        ax.hist(p_pos[y_true == lab], bins=bins, alpha=0.65, color=COLORS[lab], label=f"true {lab}",
                edgecolor="white")
    ax.axvspan(0.4, 0.6, color="#e5e4e0", zorder=0, label="neutral band")
    ax.set_xlabel("P(positive) given by the network")
    ax.set_ylabel("Number of test reviews")
    ax.set_title("The network is usually very sure (near 0 or 1)", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "ex2_probability_histogram.png", dpi=150)
    plt.close(fig)

    print(f"\nFiles saved in: {OUT}")


if __name__ == "__main__":
    main()
