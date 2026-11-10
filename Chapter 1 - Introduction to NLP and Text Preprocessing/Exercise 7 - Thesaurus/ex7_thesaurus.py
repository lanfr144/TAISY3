"""
Chapter 1 - Exercise 7 : Thesaurus - Synonyms & Antonyms with WordNet
======================================================================

KEY IDEAS FOR A JUNIOR
----------------------
WordNet is a big English dictionary organised for computers (included in NLTK).

* SYNSET  = "synonym set" = ONE MEANING of a word.
            "car" has 5 synsets: the automobile, a railway carriage, the
            cabin of a lift, ...  A synset has a name like  car.n.01
                                                        word . part-of-speech . number
            (n = noun, v = verb, a / s = adjective, r = adverb)
* LEMMA   = one way of writing a meaning.  The synset car.n.01 contains the
            lemmas: car, auto, automobile, machine, motorcar.
            -> all lemmas of all synsets of a word = its SYNONYMS
* ANTONYM = the opposite meaning.  In WordNet antonyms are attached to a
            LEMMA (not to a synset):  lemma.antonyms()

                word ──> synsets (meanings) ──> lemmas (synonyms) ──> antonyms

SETUP (once)
------------
    pip install nltk
(the script downloads the WordNet data automatically the first time)

HOW TO RUN
----------
    python ex7_thesaurus.py

OUTPUT
------
    * synonyms / antonyms / bonus / reflection in the terminal
    * outputs/ex7_thesaurus_overview.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

try:
    import nltk
    from nltk.corpus import wordnet as wn
except ImportError:
    sys.exit("NLTK is not installed.  Run:  pip install nltk")

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1 - make sure the WordNet corpus is installed
# ---------------------------------------------------------------------------
def ensure_wordnet():
    """Download WordNet only if it is not already on the computer."""
    for package in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{package}")
        except LookupError:
            print(f"Downloading NLTK '{package}' (only the first time)...")
            nltk.download(package, quiet=True)


# ---------------------------------------------------------------------------
# Step 2 - synonyms
# ---------------------------------------------------------------------------
def get_synonyms(word: str) -> set:
    """All unique synonyms of `word`.

    1. wn.synsets(word)      -> every possible meaning
    2. synset.lemmas()        -> every word that expresses this meaning
    3. a set()                -> duplicates disappear automatically
    WordNet writes multi-word expressions with "_" (ice_cream) -> we use a space.
    """
    synonyms = set()
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower():          # the word is not its own synonym
                synonyms.add(name)
    return synonyms


# ---------------------------------------------------------------------------
# Step 3 - antonyms
# ---------------------------------------------------------------------------
def get_antonyms(word: str) -> set:
    """All unique antonyms of `word` (antonyms hang on the lemmas)."""
    antonyms = set()
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            for ant in lemma.antonyms():
                antonyms.add(ant.name().replace("_", " "))
    return antonyms


# ---------------------------------------------------------------------------
# Bonus - do two words share a meaning?
# ---------------------------------------------------------------------------
def same_meaning(word1: str, word2: str) -> bool:
    """True if the two words have at least one synset (meaning) in common.

    The "&" between two sets = the INTERSECTION (elements present in both).
    """
    shared = set(wn.synsets(word1)) & set(wn.synsets(word2))
    return len(shared) > 0


def short(words, limit=12):
    """Readable printing of a (possibly long) set."""
    words = sorted(words)
    txt = ", ".join(words[:limit])
    return "{" + txt + (f", ... +{len(words) - limit} more" if len(words) > limit else "") + "}"


def main():
    ensure_wordnet()

    # Step 2 test ------------------------------------------------------------
    print("=" * 80)
    print("STEP 2 - Synonyms")
    print("=" * 80)
    syn_words = ["happy", "fast", "small", "car"]
    for w in syn_words:
        print(f"  {w:<6} -> {short(get_synonyms(w))}")

    # Step 3 test ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3 - Antonyms")
    print("=" * 80)
    ant_words = ["hot", "happy", "fast", "small"]
    for w in ant_words:
        print(f"  {w:<6} -> {short(get_antonyms(w))}")

    # Step 4 - readable combined output ----------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4 - Combined view")
    print("=" * 80)
    all_words = ["happy", "fast", "small", "car", "hot"]
    stats = {}
    for w in all_words:
        syns, ants = get_synonyms(w), get_antonyms(w)
        stats[w] = (len(wn.synsets(w)), len(syns), len(ants))
        print(f"  {w} -> Synonyms: {short(syns, 6)} | Antonyms: {short(ants, 6)}")

    # Bonus ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BONUS - same_meaning(word1, word2)")
    print("=" * 80)
    for a, b in [("car", "automobile"), ("buy", "purchase"), ("car", "dog")]:
        shared = set(wn.synsets(a)) & set(wn.synsets(b))
        print(f"  same_meaning('{a}', '{b}') -> {same_meaning(a, b)}"
              + (f"   shared meaning: {[s.name() for s in shared]}" if shared else ""))

    # Reflection ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("REFLECTION - the word 'bank'")
    print("=" * 80)
    bank = wn.synsets("bank")
    print(f"  'bank' has {len(bank)} different meanings in WordNet, for example:")
    for s in bank[:6]:
        print(f"    {s.name():<14} {s.definition()}")
    print(f"\n  All synonyms mixed together: {short(get_synonyms('bank'), 15)}")
    print("""
  WHY CAN A THESAURUS BE MISLEADING WITHOUT CONTEXT?
  * One word, many meanings (polysemy): 'bank' = a river side, a company that
    keeps money, a row of switches, an aircraft turning ...
  * get_synonyms() mixes ALL the meanings: it may propose 'depository financial
    institution' for "we had a picnic on the river bank" -> nonsense.
  * Even the part of speech changes the meaning: to bank (verb) = to tilt a plane.
  * The fix in NLP is called WORD SENSE DISAMBIGUATION: look at the surrounding
    words first, choose ONE synset, then take only its synonyms.
    (NLTK even has a simple version: from nltk.wsd import lesk)""")

    try:
        from nltk.wsd import lesk
        for sentence in ["I deposited money at the bank", "We sat on the bank of the river"]:
            sense = lesk(sentence.split(), "bank")
            if sense is not None:
                print(f"\n  lesk('{sentence}') -> {sense.name()}: {sense.definition()}")
        print("  (lesk is a very simple algorithm: it is not always right, but it shows the idea.)")
    except Exception:  # lesk is only a demonstration; never block the exercise
        pass

    # ------------------------------------------------------------------------
    # GRAPH: meanings, synonyms and antonyms per word + bank meanings by POS
    # ------------------------------------------------------------------------
    BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.7, 1]})

    x = range(len(all_words))
    w = 0.27
    series = [("Meanings (synsets)", 0, BLUE), ("Synonyms", 1, ORANGE), ("Antonyms", 2, AQUA)]
    for k, (label, idx, color) in enumerate(series):
        vals = [stats[word][idx] for word in all_words]
        bars = ax1.bar([i + (k - 1) * w for i in x], vals, w, label=label, color=color)
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     str(int(bar.get_height())), ha="center", fontsize=8, color=GREY)
    ax1.set_xticks(list(x), all_words)
    ax1.set_ylabel("Count in WordNet")
    ax1.set_title("More meanings -> more (and more mixed) synonyms", loc="left", weight="bold")
    ax1.legend(frameon=False)

    pos_names = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}
    pos_count = {}
    for s in bank:
        pos_count[pos_names[s.pos()]] = pos_count.get(pos_names[s.pos()], 0) + 1
    ax2.bar(list(pos_count), list(pos_count.values()), color=BLUE, width=0.5)
    for i, v in enumerate(pos_count.values()):
        ax2.text(i, v + 0.15, str(v), ha="center", color=GREY)
    ax2.set_ylabel("Number of meanings")
    ax2.set_title("'bank': meanings by part of speech", loc="left", weight="bold")

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Chapter 1 - Exercise 7: WordNet thesaurus", weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "ex7_thesaurus_overview.png", dpi=150)
    plt.close(fig)
    print(f"\nGraph saved to: {OUT / 'ex7_thesaurus_overview.png'}")


if __name__ == "__main__":
    main()
