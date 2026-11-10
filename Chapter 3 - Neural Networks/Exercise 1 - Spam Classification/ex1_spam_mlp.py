"""
Chapter 3 - Exercise 1 : Spam classification with a Neural Network (MLP)
==========================================================================

SAME PROBLEM AS CHAPTER 2 - EXERCISE 3, NEW MODEL
-------------------------------------------------
We replace Naive Bayes by a NEURAL NETWORK and compare.

WHAT IS AN MLP (Multi-Layer Perceptron)?
----------------------------------------

      INPUT layer            HIDDEN layer             OUTPUT layer
   (1 neuron per word)    (here 100 neurons)        (spam or ham)

     "free"   o ───┐
     "win"    o ───┼────>  o  o  o  ...  o  ─────>      o  P(spam)
     "meeting"o ───┘
       ...

* Every arrow has a WEIGHT (a number).  A neuron computes
        output = activation( sum(input x weight) + bias )
* The ACTIVATION function adds non-linearity:
        relu(x)     = max(0, x)          (default, fast)
        logistic(x) = 1 / (1 + e^-x)     (the "sigmoid", output between 0 and 1)
* TRAINING = adjusting all the weights, little by little, so that the
  predictions get closer to the true labels.  One pass over the training
  data = one EPOCH.  The LOSS measures the error: it should go DOWN.

scikit-learn: MLPClassifier(hidden_layer_sizes=(100,), activation="relu", max_iter=...)
    hidden_layer_sizes=(100,)    -> 1 hidden layer of 100 neurons
    hidden_layer_sizes=(50, 20)  -> 2 hidden layers: 50 then 20 neurons

A WORD OF WARNING
-----------------
The dataset has only 48 e-mails, but the network has thousands of weights
(about 190 words x 100 neurons = 19 000 - the script prints the exact number).
Neural networks NEED lots of data; on tiny datasets they easily OVERFIT
(learn the training set by heart).
Naive Bayes is often just as good here: the right tool depends on the data!

HOW TO RUN
----------
    python ex1_spam_mlp.py

OUTPUT (outputs/)
-----------------
    ex1_loss_curves.png          training loss per epoch (Count vs TF-IDF)
    ex1_confusion_matrices.png
    ex1_models_comparison.png    MLP vs Naive Bayes, Count vs TF-IDF
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
DATA_FILE = HERE / "Ch2_Ex3_Spam.csv"
RANDOM_STATE = 42
BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#52514e"

# On such a small dataset the MLP may stop before it "officially" converges:
# we hide that warning to keep the output readable (the loss curve shows it anyway).
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def make_mlp():
    """The neural network used in this exercise (the exercise sheet's settings)."""
    return MLPClassifier(hidden_layer_sizes=(100,),   # 1 hidden layer, 100 neurons
                         activation="relu",
                         solver="adam",               # the algorithm that updates the weights
                         max_iter=300,                # max number of epochs
                         random_state=RANDOM_STATE)   # same starting weights every run


def main():
    # 1. Load -------------------------------------------------------------------
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} e-mails: {df['label'].value_counts().to_dict()}")

    # 3. Split -------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=RANDOM_STATE, stratify=df["label"])
    print(f"Train: {len(X_train)}   Test: {len(X_test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    configs = [
        ("MLP + CountVectorizer", CountVectorizer, make_mlp, BLUE),
        ("MLP + TfidfVectorizer", TfidfVectorizer, make_mlp, ORANGE),
        ("Naive Bayes + CountVectorizer (Ch.2)", CountVectorizer, MultinomialNB, AQUA),
        ("Naive Bayes + TfidfVectorizer (Ch.2)", TfidfVectorizer, MultinomialNB, YELLOW),
    ]

    results = []
    for name, Vec, Model, color in configs:
        # 2 + 4. vectorize + train  (pipeline = vectorizer learns on TRAIN only)
        pipe = make_pipeline(Vec(), Model())
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        # 5. measure
        cv_scores = cross_val_score(make_pipeline(Vec(), Model()), df["text"], df["label"], cv=cv)
        n_weights = None
        model = pipe[-1]
        if isinstance(model, MLPClassifier):
            n_weights = sum(c.size for c in model.coefs_) + sum(b.size for b in model.intercepts_)
        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)
        if n_weights:
            print(f"Network: {pipe[0].transform(['x']).shape[1]} inputs -> 100 hidden -> 1 output"
                  f" = {n_weights:,} weights to learn, trained for {model.n_iter_} epochs")
        print(classification_report(y_test, y_pred, zero_division=0))
        print(f"5-fold cross-validation accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})")
        results.append({"name": name, "pipe": pipe, "y_pred": y_pred, "color": color,
                        "test_accuracy": accuracy_score(y_test, y_pred),
                        "test_f1_spam": f1_score(y_test, y_pred, pos_label="spam", zero_division=0),
                        "cv_accuracy": cv_scores.mean(), "cv_std": cv_scores.std()})

    table = pd.DataFrame(results).set_index("name")[["test_accuracy", "test_f1_spam", "cv_accuracy"]]
    print("\n" + "=" * 70)
    print("COMPARISON (challenge: Count vs TF-IDF, and MLP vs Naive Bayes)")
    print("=" * 70)
    print((table * 100).round(1).astype(str).add(" %").to_string())
    mlp_best = max(results[:2], key=lambda r: r["cv_accuracy"])
    nb_best = max(results[2:], key=lambda r: r["cv_accuracy"])
    print(f"\nBest MLP        : {mlp_best['name']} ({mlp_best['cv_accuracy']:.0%} cross-validated)")
    print(f"Best Naive Bayes: {nb_best['name']} ({nb_best['cv_accuracy']:.0%} cross-validated)")
    if mlp_best["cv_accuracy"] > nb_best["cv_accuracy"] + 0.02:
        print("-> the neural network wins, even on this small dataset.")
    elif nb_best["cv_accuracy"] > mlp_best["cv_accuracy"] + 0.02:
        print("-> Naive Bayes wins: with 48 e-mails the network does not have enough data.")
    else:
        print("-> practically a draw: the neural network brings no real gain on 48 e-mails,")
        print("   but it is much slower to train and much harder to explain.")

    # ------------------------------------------------------------------------
    # GRAPHS
    # ------------------------------------------------------------------------
    # 1. Loss curves: the error of the network after every epoch
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results[:2]:
        ax.plot(r["pipe"][-1].loss_curve_, color=r["color"], linewidth=2, label=r["name"])
    ax.set_xlabel("Epoch (one pass over the training e-mails)")
    ax.set_ylabel("Training loss (log-loss)")
    ax.set_title("The MLP learns: its error goes down at every epoch", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "ex1_loss_curves.png", dpi=150)
    plt.close(fig)

    # 2. Confusion matrices of the two MLPs
    labels = sorted(df["label"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, r in zip(axes, results[:2]):
        cm = confusion_matrix(y_test, r["y_pred"], labels=labels)
        ax.imshow(cm, cmap="Blues", vmin=0)
        ax.set_xticks(range(2), [f"pred. {l}" for l in labels])
        ax.set_yticks(range(2), [f"true {l}" for l in labels])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                        color="white" if cm[i, j] > cm.max() / 2 else "#0b0b0b")
        ax.set_title(f"{r['name']}\ntest accuracy {r['test_accuracy']:.0%}", loc="left", weight="bold")
    fig.suptitle("Chapter 3 - Exercise 1: confusion matrices on the test set", weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "ex1_confusion_matrices.png", dpi=150)
    plt.close(fig)

    # 3. All four models
    fig, ax = plt.subplots(figsize=(11, 5))
    y = np.arange(len(results))
    vals = [r["cv_accuracy"] * 100 for r in results]
    ax.barh(y, vals, xerr=[r["cv_std"] * 100 for r in results], color=[r["color"] for r in results],
            height=0.6, error_kw={"ecolor": GREY, "capsize": 4})
    for yi, v, r in zip(y, vals, results):
        ax.text(2, yi, f"{v:.0f} %  (test split: {r['test_accuracy']:.0%})", va="center",
                color="white", weight="bold", fontsize=9)
    ax.set_yticks(y, [r["name"] for r in results])
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("5-fold cross-validated accuracy (%)  - error bar = spread over the folds")
    ax.set_title("Neural network vs Naive Bayes on the spam dataset", loc="left", weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "ex1_models_comparison.png", dpi=150)
    plt.close(fig)

    print(f"\nGraphs saved in: {OUT}")


if __name__ == "__main__":
    main()
