"""
Chapter 1 - Exercise 3 : Tokenization and Stopwords (spaCy)
============================================================

KEY IDEAS FOR A JUNIOR
----------------------
* TOKEN      = one "piece" of text: a word, a number, a punctuation sign.
               "I can't go!" -> ["I", "ca", "n't", "go", "!"]
               (spaCy splits "can't" into "ca" + "n't" because it means "can not")
* TOKENIZER  = the program that cuts the text into tokens.  Splitting on
               spaces is NOT enough: "go!" must give "go" and "!".
* STOPWORD   = a very frequent word that carries little meaning on its own
               ("the", "is", "and", "of" ...).  We often remove them so the
               model focuses on the important words.
* LOWERCASING = "Python" and "python" become the same token.

spaCy gives every token useful flags:
    token.text      the text of the token
    token.is_punct  True if it is punctuation
    token.is_stop   True if it is a stopword
    token.is_space  True if it is only whitespace / newline

SETUP (once)
------------
    pip install spacy
    python -m spacy download en_core_web_sm

HOW TO RUN
----------
    python ex3_tokenization.py                       -> asks you for a sentence
    python ex3_tokenization.py --sentence "Hi there!" -> no question asked
(If you just press Enter, an example sentence is used.)

OUTPUT
------
    * tasks 1-5 and the challenge in the terminal
    * outputs/ex3_token_filtering.png
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

DEFAULT_SENTENCE = "The students aren't sure why the NLP course, given on Monday, is so exciting!"

PARAGRAPH = """
Natural Language Processing is a fascinating field of Artificial Intelligence.
It allows computers to understand, interpret, and manipulate human language.
By removing stopwords and punctuation, we can extract the essential terms of a text.
The essential terms are often enough to understand what a text is about!
"""


def load_model(name="en_core_web_sm"):
    """Load the small English model, with a clear message if it is missing."""
    try:
        return spacy.load(name)
    except OSError:
        sys.exit(f"The spaCy model '{name}' is missing.  Run:  python -m spacy download {name}")


def ask_sentence(cli_sentence):
    """Task 1 asks the USER for a sentence.  --sentence skips the question."""
    if cli_sentence:
        return cli_sentence
    try:
        typed = input("Enter a sentence (or press Enter for an example): ").strip()
    except EOFError:  # no keyboard (e.g. run from an automated tool)
        typed = ""
    return typed or DEFAULT_SENTENCE


def title(text):
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Tokenization and stopwords with spaCy")
    parser.add_argument("--sentence", help="sentence to analyse (otherwise you are asked)")
    args = parser.parse_args()

    nlp = load_model()
    sentence = ask_sentence(args.sentence)
    doc = nlp(sentence)          # <- spaCy does all the work here
    print(f"\nSentence: {sentence!r}")

    # Task 1 - Tokenize: print each token on a new line ---------------------
    title("Task 1 - Each token on a new line")
    for token in doc:
        print(f"  [{token.i:2d}] {token.text}")

    # Task 2 - Count tokens ---------------------------------------------------
    # A Doc behaves like a list of tokens, so len() gives the number of tokens.
    title("Task 2 - Count tokens")
    n_tokens = len(doc)
    n_words_by_split = len(sentence.split())
    print(f"  spaCy tokens        : {n_tokens}")
    print(f"  simple .split()     : {n_words_by_split}   <- fewer, because '!' or ',' stay glued to words")

    # Task 3 - Remove punctuation ---------------------------------------------
    title("Task 3 - Tokens that are NOT punctuation")
    no_punct = [t for t in doc if not t.is_punct and not t.is_space]
    print("  " + " | ".join(t.text for t in no_punct))

    # Task 4 - Remove stopwords -----------------------------------------------
    title("Task 4 - Tokens that are NOT stopwords")
    no_stop = [t for t in doc if not t.is_stop and not t.is_space]
    print("  " + " | ".join(t.text for t in no_stop))
    removed = [t.text for t in doc if t.is_stop]
    print(f"  (stopwords removed: {removed})")
    print(f"  spaCy knows {len(nlp.Defaults.stop_words)} English stopwords.")

    # Task 5 - Lowercasing ------------------------------------------------------
    title("Task 5 - Tokens in lowercase")
    print("  " + " | ".join(t.text.lower() for t in doc if not t.is_space))

    # Challenge - unique words of a paragraph ----------------------------------
    # Filter:  not punctuation, not stopword, not space  -> lowercase -> set()
    # A set() keeps every value only once = "unique words".
    title("Challenge - Unique words of a paragraph")
    pdoc = nlp(PARAGRAPH)
    kept = [t.text.lower() for t in pdoc
            if not t.is_punct and not t.is_stop and not t.is_space]
    unique_words = sorted(set(kept))
    print(PARAGRAPH.strip())
    print(f"\n  {len(unique_words)} unique words: {unique_words}")
    print("\n  Words that appear more than once:",
          {w: c for w, c in Counter(kept).items() if c > 1})

    # ------------------------------------------------------------------------
    # GRAPH: how many tokens survive each filter (sentence AND paragraph)
    # ------------------------------------------------------------------------
    def filter_counts(d):
        toks = [t for t in d if not t.is_space]
        return {
            "All tokens": len(toks),
            "Without punctuation": sum(not t.is_punct for t in toks),
            "Without stopwords": sum(not t.is_stop for t in toks),
            "Without both": sum(not t.is_punct and not t.is_stop for t in toks),
        }

    sent_c, para_c = filter_counts(doc), filter_counts(pdoc)
    labels = list(sent_c)
    x = range(len(labels))
    w = 0.38
    BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#52514e"

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar([i - w / 2 for i in x], sent_c.values(), w, color=BLUE, label="Your sentence")
    b2 = ax.bar([i + w / 2 for i in x], para_c.values(), w, color=ORANGE, label="Challenge paragraph")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    str(int(bar.get_height())), ha="center", color=GREY, fontsize=9)
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Number of tokens")
    ax.set_title("How many tokens are left after each filter?", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "ex3_token_filtering.png", dpi=150)
    plt.close(fig)
    print(f"\nGraph saved to: {OUT / 'ex3_token_filtering.png'}")
    print("Notice: removing stopwords removes MANY tokens -> the text keeps only its 'content' words.")


if __name__ == "__main__":
    main()
