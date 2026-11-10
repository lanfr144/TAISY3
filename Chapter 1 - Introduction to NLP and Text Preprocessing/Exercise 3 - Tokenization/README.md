# Chapter 1 – Exercise 3: Tokenization and Stopwords (spaCy)

[⬅ Back to the chapter](../README.md) · [🏠 Main page](../../README.md)

📄 **Exercise sheet:** [Exercise_3-Tokenization.docx](Exercise_3-Tokenization.docx)

## Goal

Ask the user for a sentence, cut it into tokens with spaCy, count them, remove punctuation and stopwords, lowercase them, and list the unique words of a paragraph.

## Key ideas (for a junior)

* **Token** – one piece of text: a word, a number or a punctuation sign. spaCy splits `can't` into `ca` + `n't`.
* **Why not `split()`?** – `"go!".split()` keeps `go!` as one piece; a tokenizer gives `go` and `!`.
* **Stopwords** – very frequent words with little meaning (*the, is, and*). spaCy knows ~300 of them (`token.is_stop`).
* **Token flags** – `token.is_punct`, `token.is_stop`, `token.is_space`, `token.text.lower()`.
* **`set()`** – keeps every value once → "unique words".

## What the script does

The script `ex3_tokenization.py` follows the exercise sheet step by step; every step is commented in the code.

| # | Task | How |
|---|---|---|
| 1 | Tokenize a sentence typed by the user | `input()` (or `--sentence`), one token per line |
| 2 | Count tokens | `len(doc)` (compared with `split()`) |
| 3 | Remove punctuation | `not token.is_punct` |
| 4 | Remove stopwords | `not token.is_stop` |
| 5 | Lowercasing | `token.text.lower()` |
| 6 | Challenge: unique words of a paragraph | filter + lowercase + `sorted(set(...))` |

## How to run

In a terminal (Command Prompt, PowerShell or the VS Code terminal):

```bash
cd "Chapter 1 - Introduction to NLP and Text Preprocessing/Exercise 3 - Tokenization"
python ex3_tokenization.py --sentence "Hi there, how are you?"
```
You can also run it from any other folder: the script always finds its data and writes
into its own `outputs/` folder.  First time? Run `setup_environment.bat` in the main folder.

Needs: `pip install spacy` and `python -m spacy download en_core_web_sm`.

## Results

> **Graphs:** this exercise needs `spaCy`.  Run the script once on your computer (after `setup_environment.bat`): the graphs listed below will appear in the `outputs/` folder.

## Common mistakes

* Forgetting `token.is_space`: the newlines of a multi-line paragraph become "words".
* Testing stopwords after lowercasing *your own* string – use spaCy's flags on the token instead.

## Files

- `Exercise_3-Tokenization.docx` – the exercise sheet (from Moodle)
- `ex3_tokenization.py` – the solution, with comments
- `outputs/` – created by the script:
  - ex3_token_filtering.png – tokens left after each filter, for your sentence and the paragraph
