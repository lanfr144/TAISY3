"""
Chapter 1 - Exercise 2 : Regular Expressions on Social Media posts
==================================================================

WHY THIS EXERCISE?
------------------
Tweets and posts are the "dirtiest" text you will meet: mentions, hashtags,
links, emojis, slang, repeated letters ("happppyyyy"), numbers everywhere.
A model does not care about the exact URL or the exact phone number, so we
replace them by a *placeholder token* such as <URL> or <PHONE>.  This keeps
the information "there was a link here" while removing the noise.

THE ORDER OF THE STEPS MATTERS (very important!)
------------------------------------------------
    * e-mails  BEFORE mentions   -> "email_me@domain.co.uk" contains an "@"
    * URLs     BEFORE numbers    -> "bit.ly/3xYzQe" contains digits
    * phones   BEFORE numbers    -> "+44 20 7946 0958" would become <NUM> <NUM>...
    * slang    BEFORE numbers    -> "4" (= for) and "2moro" contain digits
    * operators BEFORE emojis... no conflict, but ">>>" must be tested before ">>"
If you change the order, you will see the results get worse: try it!

PANDAS METHODS REQUIRED BY THE SHEET
------------------------------------
    .str.replace(pattern, replacement, regex=True)
    .str.extract(pattern)      -> first match
    .str.findall(pattern)      -> all matches (list)

HOW TO RUN
----------
    python ex2_regex_social_media.py

OUTPUT
------
    * a table per step in the terminal
    * outputs/ex2_cleaned_posts.csv
    * outputs/ex2_cleaning_overview.png
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Windows terminals sometimes cannot print emojis -> force UTF-8 output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 200)

data = {
    "post": [
        "RT @user123: LOL!!! Can't believe it... 😂😂 http://bit.ly/3xYzQe",
        "Weather in NYC is CRAZY—100°F!! 🔥🔥 #climatechange",
        "New iPhone14 released!!! Pre-order at https://apple.com 📱💸",
        "Sooo happppyyyyy to get 1st placeeeee!!! 🏆🏆",
        "@friend Thanks 4 ur help!!! cu 2moro ;)",
        "Bitcoin hits $35,000!! 🚀🚀 #crypto #btc",
        "WTF??? Java >>> Python?????!! lololol",
        "Contact me: email_me@domain.co.uk OR +44 20 7946 0958",
    ]
}
df = pd.DataFrame(data)


def step(title, before, after):
    """Print only the rows that changed, so you see the effect of a step."""
    print(f"\n--- {title} ---")
    changed = before != after
    if not changed.any():
        print("   (no change)")
    for b, a in zip(before[changed], after[changed]):
        print(f"   {b}\n   -> {a}")


# ---------------------------------------------------------------------------
# Regex patterns (defined once, reused below)
# ---------------------------------------------------------------------------
RETWEET = r"^RT\s+@\w+:\s*"                     # "RT @user123: " at the start
EMAIL = r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"      # user@domain.co.uk
MENTION = r"(?<![\w.])@\w+"                      # @friend, but NOT the @ of an email
HASHTAG = r"#(\w+)"                              # the group (\w+) = the word only
URL = r"https?://\S+"
PHONE = r"\+\d{1,3}(?:[\s-]?\d{2,4}){2,4}"       # +44 20 7946 0958
# Emojis live in a few Unicode "blocks". This class covers the common ones.
EMOJI = (
    "["
    "\U0001F300-\U0001F5FF"  # symbols & pictographs (🔥 🏆 📱 💸)
    "\U0001F600-\U0001F64F"  # faces (😂)
    "\U0001F680-\U0001F6FF"  # transport & map (🚀)
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "☀-➿"          # misc symbols & dingbats (☀ ✓)
    "]+"
)
ELONGATED = r"(\w)\1{2,}"     # a letter followed by itself 2+ more times (3+ in a row)
NUMBER = r"\$?\d+(?:[.,]\d+)*(?:°[FC]|st|nd|rd|th)?"   # 100°F  $35,000  1st  14
CAPITALS = r"\b[A-Z]{2,}\b"   # words written fully in capitals (NYC, WTF, LOL)

# ===========================================================================
# EXTRACTION columns (done on the ORIGINAL post)
# ===========================================================================
# Task 2: mentions  -> list of all @names
df["mentions"] = df["post"].str.findall(MENTION)
# Task 3: hashtags  -> list of all #tags (we keep the # here)
df["hashtags"] = df["post"].str.findall(r"#\w+")
# Task 13: capitalized words -> NYC, WTF, LOL, RT, OR ...
df["capitalized_words"] = df["post"].str.findall(CAPITALS)
# Bonus: first e-mail with .str.extract (shows the difference with findall)
df["first_email"] = df["post"].str.extract(f"({EMAIL})", expand=False)

print("=" * 90)
print("EXTRACTED COLUMNS")
print("=" * 90)
print(df[["post", "mentions", "hashtags", "capitalized_words", "first_email"]].to_string())

# ===========================================================================
# CLEANING PIPELINE  (each step works on the result of the previous one)
# ===========================================================================
print("\n" + "=" * 90)
print("CLEANING PIPELINE - only the rows that change are shown")
print("=" * 90)

s = df["post"].copy()

# Task 1: remove retweet markers -------------------------------------------------
new = s.str.replace(RETWEET, "", regex=True)
step("Task 1: remove 'RT @user:'", s, new); s = new

# Task 9: e-mails -> <EMAIL>   (before the mentions, see the docstring) -----------
new = s.str.replace(EMAIL, "<EMAIL>", regex=True)
step("Task 9: e-mails -> <EMAIL>", s, new); s = new

# Task 10: international phone numbers -> <PHONE> ---------------------------------
new = s.str.replace(PHONE, "<PHONE>", regex=True)
step("Task 10: phones -> <PHONE>", s, new); s = new

# Task 5: URLs -> <URL> -------------------------------------------------------------
new = s.str.replace(URL, "<URL>", regex=True)
step("Task 5: URLs -> <URL>", s, new); s = new

# Remaining mentions: the sheet only asks to EXTRACT them, so we keep the
# word but remove the "@" (a model can still learn from "friend").
new = s.str.replace(MENTION, lambda m: m.group(0)[1:], regex=True)
step("Mentions: '@friend' -> 'friend'", s, new); s = new

# Task 4: hashtags -> just the word ------------------------------------------------
# r"\1" means "put back what group 1 (\w+) matched".
new = s.str.replace(HASHTAG, r"\1", regex=True)
step("Task 4: '#crypto' -> 'crypto'", s, new); s = new

# Task 6: emojis -> <EMOJI>   (😂😂 counts as ONE emoji token thanks to "+") -----
new = s.str.replace(EMOJI, " <EMOJI> ", regex=True)
step("Task 6: emojis -> <EMOJI>", s, new); s = new

# Task 12: comparison operators -> words (longest first: >>> before >>) ----------
for op, words in [(">>>", " much greater than "), (">>", " greater than "), ("<<", " less than ")]:
    new = s.str.replace(op, words, regex=False)
    step(f"Task 12: '{op}' -> '{words.strip()}'", s, new); s = new

# Task 7: elongated words ---------------------------------------------------------------
# The sheet says "reduce >3 repeating characters to 2".  We reduce 3 or more
# to 2 ("Sooo" -> "Soo", "happppyyyyy" -> "happyy").  A regex cannot know that
# "happyy" should be "happy": that needs a dictionary / spell-checker.  We add a
# tiny dictionary to show the idea - in a real project you would use a library
# such as `symspellpy` or `textblob`.
new = s.str.replace(ELONGATED, r"\1\1", regex=True)
step("Task 7a: 3+ repeated letters -> 2", s, new); s = new
SPELL_FIX = {r"\bsoo\b": "so", r"\bhappyy\b": "happy", r"\bplacee\b": "place"}
for wrong, right in SPELL_FIX.items():
    new = s.str.replace(wrong, right, regex=True, flags=re.IGNORECASE)
    if (new != s).any():
        step(f"Task 7b: dictionary fix {wrong} -> {right}", s, new)
    s = new

# Task 11: expand internet slang (BEFORE numbers, because of "4" and "2moro") ----
# \b = word boundary, so "4" inside "iPhone14" is NOT replaced.
SLANG = {
    r"\bur\b": "your",
    r"\b4\b": "for",
    r"\b2moro\b": "tomorrow",
    r"\bcu\b": "see you",
    r"\b(?:lol)+(?:o?l)*\b": "laughing out loud",   # lol, LOL, lololol
    r"\bwtf\b": "what the heck",
}
for pattern, meaning in SLANG.items():
    new = s.str.replace(pattern, meaning, regex=True, flags=re.IGNORECASE)
    if (new != s).any():
        step(f"Task 11: slang {pattern} -> '{meaning}'", s, new)
    s = new

# Task 8: numbers -> <NUM>  ("100°F", "$35,000", "1st", "14") ---------------------
new = s.str.replace(NUMBER, "<NUM>", regex=True)
step("Task 8: numbers -> <NUM>", s, new); s = new

# Final touch: remove leftover punctuation (!!! ??? ... ;) ) but KEEP the
# placeholders <URL> etc. and apostrophes (can't).  Then lowercase the words
# while keeping placeholders in capitals so they stand out.
s = s.str.replace(r"(?!<[A-Z]+>)[^\w\s<>']", " ", regex=True)
s = s.str.replace(r"(?<!<)\b(\w+)\b(?!>)", lambda m: m.group(1).lower(), regex=True)
s = s.str.replace(r"\s+", " ", regex=True).str.strip()

# Task 15: final cleaned column -------------------------------------------------------
df["cleaned_post"] = s
# Task 14: word count after cleaning ---------------------------------------------------
df["word_count_before"] = df["post"].str.split().str.len()
df["word_count"] = df["cleaned_post"].str.split().str.len()

print("\n" + "=" * 90)
print("FINAL RESULT (Task 14 + 15)")
print("=" * 90)
for _, row in df.iterrows():
    print(f"\nORIGINAL : {row['post']}")
    print(f"CLEANED  : {row['cleaned_post']}   ({row['word_count']} words)")

df.to_csv(OUT / "ex2_cleaned_posts.csv", index=False, encoding="utf-8")

# ===========================================================================
# GRAPH
# ===========================================================================
placeholders = ["<URL>", "<EMOJI>", "<NUM>", "<EMAIL>", "<PHONE>"]
counts = {p: int(df["cleaned_post"].str.count(re.escape(p)).sum()) for p in placeholders}

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#52514e"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.6, 1]})

x = range(len(df))
w = 0.38
ax1.bar([i - w / 2 for i in x], df["word_count_before"], w, label="Before cleaning", color=GREY)
ax1.bar([i + w / 2 for i in x], df["word_count"], w, label="After cleaning", color=BLUE)
ax1.set_xticks(list(x), [f"Post {i + 1}" for i in x])
ax1.set_ylabel("Number of words (split on spaces)")
ax1.set_title("Words per post, before vs after cleaning", loc="left", weight="bold")
ax1.legend(frameon=False)

ax2.barh(list(counts), list(counts.values()), color=ORANGE, height=0.6)
for y, v in enumerate(counts.values()):
    ax2.text(v + 0.1, y, str(v), va="center", color=GREY)
ax2.invert_yaxis()
ax2.set_xlabel("How many times the placeholder appears")
ax2.set_title("Placeholders inserted", loc="left", weight="bold")

for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y" if ax is ax1 else "x", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)

fig.suptitle("Chapter 1 - Exercise 2: cleaning social-media posts", weight="bold")
fig.tight_layout()
fig.savefig(OUT / "ex2_cleaning_overview.png", dpi=150)
plt.close(fig)

print(f"\nResults saved to : {OUT / 'ex2_cleaned_posts.csv'}")
print(f"Graph saved to   : {OUT / 'ex2_cleaning_overview.png'}")
