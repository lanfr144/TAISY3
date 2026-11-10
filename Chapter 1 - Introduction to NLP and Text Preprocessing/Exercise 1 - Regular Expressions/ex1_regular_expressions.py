"""
Chapter 1 - Exercise 1 : Regular Expressions
=============================================

WHAT YOU WILL LEARN
-------------------
A *regular expression* (regex) is a small pattern language that describes
"what a piece of text looks like".  Instead of writing loops that test every
character, you describe the shape once and let Python find / replace it.

In NLP, regex is the very first cleaning tool: before a model can learn from
text we usually lowercase it, remove noise (punctuation, digits, symbols) and
pull out useful pieces (emails, dates, URLs ...).

THE REGEX VOCABULARY USED BELOW (cheat sheet)
---------------------------------------------
    .        any character            \\d   a digit 0-9
    \\w       a "word" char [A-Za-z0-9_]   \\s   a whitespace (space, tab, newline)
    \\b       a word boundary          [abc] one of a, b or c
    [^abc]   anything EXCEPT a,b,c    +    one or more      *  zero or more
    {n}      exactly n times          {n,} n or more times  ?  optional
    ( ... )  a group = the part we want to keep / extract

Tip: always write patterns as *raw strings* r"..." so that Python does not
interpret the backslashes itself (r"\\d" instead of "\\\\d").

PANDAS STRING METHODS USED
--------------------------
    df["col"].str.replace(pattern, new, regex=True)  -> substitute
    df["col"].str.extract(pattern)                    -> FIRST match (one value)
    df["col"].str.findall(pattern)                    -> ALL matches (a list)

HOW TO RUN
----------
    python ex1_regular_expressions.py

OUTPUT
------
    * every task printed in the terminal (original text -> result)
    * outputs/ex1_results.csv              (the full DataFrame)
    * outputs/ex1_regex_overview.png       (graph: matches + text length)
"""

import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# All result files go next to this script, whatever folder you run it from.
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# The dataset given in the exercise sheet
# ---------------------------------------------------------------------------
data = {
    "text": [
        "Hello World!!!",
        "Email me at student@example.com",
        "Price: $49.99 only!",
        "Call me at (123) 456-7890.",
        "The quick brown fox jumps over the lazy dog.",
        "Visit https://www.example.com now",
        "My postal code is 12345.",
        "Order ID: AB-1234-XY",
        "I love Python3 and pandas 2.0!",
        "Meeting on 2025-09-22 at 14:30.",
        "Symbols: ♥ ★ ☂ ✓",
        "Multiple     spaces here.",
        "Tabs\tand\nnewlines included.",
        "Remove digits 123456 from this text.",
        "Café Münsterländer – accented characters!",
    ]
}
df = pd.DataFrame(data)
text = df["text"]


def show(task_number, title, column, only_non_empty=False):
    """Print a task result as 'original  ->  result' so you can compare."""
    print(f"\n{'=' * 78}\nTask {task_number}: {title}\n{'=' * 78}")
    for original, result in zip(df["text"], df[column]):
        if only_non_empty and not result:   # skip rows where nothing was found
            continue
        print(f"  {original!r:48} -> {result!r}")


# ===========================================================================
# PART A - CLEANING (we transform the text)
# ===========================================================================

# 1) Lowercasing -------------------------------------------------------------
# No regex needed: "Hello" and "hello" should be the same word for a model.
df["t01_lower"] = text.str.lower()
show(1, "Lowercasing", "t01_lower")

# 2) Remove punctuation -------------------------------------------------------
# [^\w\s] = "any character that is NOT a word character and NOT a space".
# That covers ! ? . , : ; $ ( ) - and also symbols like ♥ or –.
df["t02_no_punct"] = text.str.replace(r"[^\w\s]", "", regex=True)
show(2, "Remove punctuation", "t02_no_punct")

# 3) Remove digits ------------------------------------------------------------
# \d+ = one or more digits in a row ("2025" is removed in one go).
df["t03_no_digits"] = text.str.replace(r"\d+", "", regex=True)
show(3, "Remove digits", "t03_no_digits")

# 4) Keep only letters (and spaces) --------------------------------------------
# Common mistake: [^a-zA-Z] also deletes the SPACES -> "HelloWorld".
# We keep \s in the allowed set.  NOTE: accented letters (é, ü) are not in
# a-z, so they disappear here; task 13 shows how to convert them first.
df["t04_letters_only"] = text.str.replace(r"[^A-Za-z\s]", "", regex=True)
show(4, "Keep only letters and spaces", "t04_letters_only")

# 5) Replace multiple spaces ----------------------------------------------------
# \s+ matches one OR MORE spaces/tabs/newlines -> replaced by ONE space.
# .str.strip() then removes spaces at the start/end.
df["t05_single_space"] = text.str.replace(r"\s+", " ", regex=True).str.strip()
show(5, "Normalize multiple spaces", "t05_single_space")

# ===========================================================================
# PART B - EXTRACTION (we pull information out of the text)
# ===========================================================================

# 6) Emails ---------------------------------------------------------------------
#   [\w.+-]+      the user name   (letters, digits, dot, plus, minus)
#   @             the at-sign
#   [\w-]+        the domain      (example)
#   (\.[\w-]+)+   one or more ".something" (.com, .co.uk)
EMAIL = r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
df["t06_emails"] = text.str.findall(EMAIL)
show(6, "Extract e-mail addresses", "t06_emails", only_non_empty=True)

# 7) URLs -----------------------------------------------------------------------
#   https?   "http" followed by an optional "s"
#   ://      literally
#   \S+      everything until the next whitespace
URL = r"https?://\S+"
df["t07_urls"] = text.str.findall(URL)
show(7, "Extract URLs", "t07_urls", only_non_empty=True)

# 8) Phone numbers like (123) 456-7890 --------------------------------------------
# Brackets are special in regex, so we "escape" them: \( and \)
PHONE = r"\(\d{3}\)\s?\d{3}-\d{4}"
df["t08_phones"] = text.str.findall(PHONE)
show(8, "Extract phone numbers (123) 456-7890", "t08_phones", only_non_empty=True)

# 9) Dates YYYY-MM-DD ---------------------------------------------------------------
# \b makes sure the date is not glued to other digits.
DATE = r"\b\d{4}-\d{2}-\d{2}\b"
df["t09_dates"] = text.str.findall(DATE)
show(9, "Extract dates YYYY-MM-DD", "t09_dates", only_non_empty=True)

# 10) Times HH:MM ----------------------------------------------------------------------
# A stricter pattern that only accepts real times 00:00 -> 23:59:
#   ([01]\d|2[0-3])  hour 00-19 or 20-23     :    [0-5]\d  minutes 00-59
TIME = r"\b(?:[01]\d|2[0-3]):[0-5]\d\b"
df["t10_times"] = text.str.findall(TIME)
show(10, "Extract times HH:MM", "t10_times", only_non_empty=True)

# 11) Words starting with a capital letter ------------------------------------------
# \b[A-Z]  a capital at the beginning of a word, then \w* = rest of the word.
df["t11_capital_words"] = text.str.findall(r"\b[A-Z]\w*")
show(11, "Words starting with a capital letter", "t11_capital_words")

# 12) Words longer than 6 characters ------------------------------------------------
# [A-Za-z]{7,} = at least 7 letters.  (We use letters, not \w, so that a
# number like "123456" is not counted as a "word".)
df["t12_long_words"] = text.str.findall(r"\b[A-Za-zÀ-ÿ]{7,}\b")
show(12, "Words longer than 6 characters", "t12_long_words")

# ===========================================================================
# PART C - UNICODE / SPECIAL CHARACTERS
# ===========================================================================


def strip_accents(s: str) -> str:
    """Café -> Cafe.

    unicodedata.normalize("NFKD") splits "é" into two pieces: "e" + an
    accent mark.  The accent marks have the category "Mn" (Mark, nonspacing);
    we keep everything else.  Symbols such as ♥ are NOT removed here: that is
    the job of task 14.
    """
    decomposed = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


# 13) Replace accented characters --------------------------------------------------
# Common mistake: deleting the accented letters ("Café" -> "Caf").
df["t13_no_accents"] = text.apply(strip_accents)
show(13, "Replace accented characters (Café -> Cafe)", "t13_no_accents")

# 14) Remove non-ASCII characters ---------------------------------------------------
# ASCII = the 128 basic characters (codes 0x00 -> 0x7F).  Anything outside
# that range (♥ ★ ☂ ✓ é –) is removed.
df["t14_ascii_only"] = text.str.replace(r"[^\x00-\x7F]+", "", regex=True)
show(14, "Remove non-ASCII characters", "t14_ascii_only")

# 15) Tokenization with regex --------------------------------------------------------
# A token = a sequence of word characters.  findall returns a Python list
# per row, which is exactly what the exercise asks.
df["t15_tokens"] = text.str.findall(r"\b\w+\b")
show(15, "Tokenization with regex (list of words)", "t15_tokens")

# ===========================================================================
# SAVE + GRAPH
# ===========================================================================
df.to_csv(OUT / "ex1_results.csv", index=False, encoding="utf-8")

# Graph 1: in how many of the 15 sentences did each extraction pattern find
# something?  -> quick sanity check that every pattern works.
extract_cols = {
    "Emails (6)": "t06_emails",
    "URLs (7)": "t07_urls",
    "Phones (8)": "t08_phones",
    "Dates (9)": "t09_dates",
    "Times (10)": "t10_times",
    "Capital words (11)": "t11_capital_words",
    "Long words (12)": "t12_long_words",
}
matches = {label: int(df[col].str.len().gt(0).sum()) for label, col in extract_cols.items()}

# Graph 2: average number of characters left after each cleaning step.
clean_cols = {
    "Original": "text",
    "No punct. (2)": "t02_no_punct",
    "No digits (3)": "t03_no_digits",
    "Letters only (4)": "t04_letters_only",
    "Single space (5)": "t05_single_space",
    "ASCII only (14)": "t14_ascii_only",
}
avg_len = {label: df[col].str.len().mean() for label, col in clean_cols.items()}

BLUE, GREY = "#2a78d6", "#52514e"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.barh(list(matches), list(matches.values()), color=BLUE, height=0.6)
for y, v in enumerate(matches.values()):
    ax1.text(v + 0.1, y, str(v), va="center", color=GREY)
ax1.invert_yaxis()
ax1.set_xlabel("Number of sentences with at least one match (out of 15)")
ax1.set_title("Extraction patterns: how many rows matched?", loc="left", weight="bold")

ax2.bar(list(avg_len), list(avg_len.values()), color=BLUE, width=0.6)
for x, v in enumerate(avg_len.values()):
    ax2.text(x, v + 0.3, f"{v:.1f}", ha="center", color=GREY)
ax2.set_ylabel("Average characters per sentence")
ax2.set_title("Each cleaning step, applied to the original text:\naverage characters left", loc="left", weight="bold")
ax2.tick_params(axis="x", rotation=25)

for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x" if ax is ax1 else "y", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)

fig.suptitle("Chapter 1 - Exercise 1: Regular expressions overview", weight="bold")
fig.tight_layout()
fig.savefig(OUT / "ex1_regex_overview.png", dpi=150)
plt.close(fig)

print(f"\nResults saved to : {OUT / 'ex1_results.csv'}")
print(f"Graph saved to   : {OUT / 'ex1_regex_overview.png'}")
