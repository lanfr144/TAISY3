"""
Chapter 1 - Exercise 4 : Lemmatization and Tokenization (spaCy)
================================================================

KEY IDEAS FOR A JUNIOR
----------------------
* LEMMA = the "dictionary form" of a word, the form you would look up.
        running, ran, runs  -> run
        children            -> child
        are, is, was        -> be
        better              -> well / good  (depends on the context)
* LEMMATIZATION reduces all variants of a word to ONE form, so a model
  understands that "cats" and "cat" are the same thing.
* LEMMATIZATION vs STEMMING
        stemming just cuts the end of a word:  "studies" -> "studi"
        lemmatization uses grammar + a dictionary: "studies" -> "study"
  spaCy does lemmatization; it first finds the PART OF SPEECH (POS) of each
  word (NOUN, VERB, ADJ ...) because "saw" (a tool, NOUN) and "saw" (past of
  see, VERB) have different lemmas.

Useful spaCy attributes:
    token.text    original word      token.lemma_  its lemma
    token.pos_    part of speech     token.is_stop / token.is_punct

SETUP (once)
------------
    pip install spacy
    python -m spacy download en_core_web_sm

HOW TO RUN
----------
    python ex4_lemmatization.py                        -> asks for a sentence
    python ex4_lemmatization.py --sentence "The mice ran away."

OUTPUT
------
    * tasks 1-3 and the challenge in the terminal
    * outputs/ex4_tokens_vs_lemmas.png
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

try:
    import spacy
except ImportError:
    sys.exit("spaCy is not installed.  Run:  pip install spacy")

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

DEFAULT_SENTENCE = "The children were running faster than the geese, and the mice hid in their houses."

PARAGRAPH = """
The researchers were studying how children learn languages.
They noticed that the kids played games while the teachers were reading stories.
Many studies have shown that playing and reading are improving the vocabulary of children.
"""


def load_model(name="en_core_web_sm"):
    try:
        return spacy.load(name)
    except OSError:
        sys.exit(f"The spaCy model '{name}' is missing.  Run:  python -m spacy download {name}")


def ask_sentence(cli_sentence):
    if cli_sentence:
        return cli_sentence
    try:
        typed = input("Enter a sentence (or press Enter for an example): ").strip()
    except EOFError:
        typed = ""
    return typed or DEFAULT_SENTENCE


def title(text):
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Lemmatization with spaCy")
    parser.add_argument("--sentence", help="sentence to analyse (otherwise you are asked)")
    args = parser.parse_args()

    nlp = load_model()
    sentence = ask_sentence(args.sentence)
    doc = nlp(sentence)

    # Task 1 - each token next to its lemma ----------------------------------
    title("Task 1 - Token  ->  lemma")
    print(f"  {'TOKEN':<14}{'LEMMA':<14}{'POS':<8}CHANGED?")
    for t in doc:
        if t.is_space:
            continue
        changed = "yes" if t.text.lower() != t.lemma_.lower() else ""
        print(f"  {t.text:<14}{t.lemma_:<14}{t.pos_:<8}{changed}")

    # Task 2 - compare tokens vs lemmas for VERBS and plural NOUNS -----------
    # We only show the tokens whose lemma is different from the token.
    title("Task 2 - What changes for verbs and plural nouns?")
    for pos, label in [("VERB", "Verbs"), ("AUX", "Auxiliary verbs (be, have)"), ("NOUN", "Nouns")]:
        pairs = [(t.text, t.lemma_) for t in doc if t.pos_ == pos and t.text.lower() != t.lemma_.lower()]
        if pairs:
            print(f"  {label}: " + ", ".join(f"{a} -> {b}" for a, b in pairs))
    print("  -> verbs go back to the infinitive, plural nouns go back to the singular.")

    # Task 3 - lemmas without stopwords and punctuation -----------------------
    title("Task 3 - Lemmas that are NOT stopwords and NOT punctuation")
    content_lemmas = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and not t.is_space]
    print("  " + " | ".join(content_lemmas))

    # Challenge - unique lemmas of a paragraph --------------------------------
    title("Challenge - Unique lemmas (lowercase, no stopwords, no punctuation)")
    pdoc = nlp(PARAGRAPH)
    tokens = [t.text.lower() for t in pdoc if not t.is_stop and not t.is_punct and not t.is_space]
    lemmas = [t.lemma_.lower() for t in pdoc if not t.is_stop and not t.is_punct and not t.is_space]
    unique_tokens, unique_lemmas = sorted(set(tokens)), sorted(set(lemmas))
    print(PARAGRAPH.strip())
    print(f"\n  {len(unique_tokens)} unique WORDS : {unique_tokens}")
    print(f"  {len(unique_lemmas)} unique LEMMAS: {unique_lemmas}")
    print(f"  -> lemmatization merged {len(unique_tokens) - len(unique_lemmas)} word variants "
          "(e.g. 'studying' + 'studies' -> 'study', 'playing' + 'played' -> 'play').")
    print("  Note: 'children' and 'kids' stay different: a lemma is a grammar form, not a synonym "
          "(synonyms = Exercise 7).")

    # ------------------------------------------------------------------------
    # GRAPH: unique words vs unique lemmas + the 10 most frequent lemmas
    # ------------------------------------------------------------------------
    BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#52514e"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.6]})

    vals = [len(unique_tokens), len(unique_lemmas)]
    ax1.bar(["Unique words", "Unique lemmas"], vals, color=[GREY, BLUE], width=0.55)
    for i, v in enumerate(vals):
        ax1.text(i, v + 0.3, str(v), ha="center", color=GREY)
    ax1.set_ylabel("Count (challenge paragraph)")
    ax1.set_title("Lemmatization shrinks the vocabulary", loc="left", weight="bold")

    top = Counter(lemmas).most_common(10)
    ax2.barh([w for w, _ in top], [c for _, c in top], color=ORANGE, height=0.6)
    ax2.invert_yaxis()
    ax2.set_xlabel("Frequency in the paragraph")
    ax2.set_title("10 most frequent content lemmas", loc="left", weight="bold")
    ax2.xaxis.get_major_locator().set_params(integer=True)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y" if ax is ax1 else "x", color="#e5e4e0", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Chapter 1 - Exercise 4: tokens vs lemmas", weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "ex4_tokens_vs_lemmas.png", dpi=150)
    plt.close(fig)
    print(f"\nGraph saved to: {OUT / 'ex4_tokens_vs_lemmas.png'}")


if __name__ == "__main__":
    main()
