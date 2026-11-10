# Chapter 1 – Exercise 4: Lemmatization and Tokenization (spaCy)

[⬅ Back to the chapter](../README.md) · [🏠 Main page](../../README.md)

📄 **Exercise sheet:** [Exercise_4-Lemmatization and Tokenization.docx](Exercise_4-Lemmatization%20and%20Tokenization.docx)

## Goal

Show every token next to its lemma, compare what changes for verbs and plural nouns, keep only the content lemmas, and list the unique lemmas of a paragraph.

## Key ideas (for a junior)

* **Lemma** – the dictionary form: *running, ran → run*, *children → child*, *are → be*.
* **Lemmatization vs stemming** – stemming cuts endings (*studies → studi*); lemmatization uses grammar and a dictionary (*studies → study*).
* **Part of speech (POS)** – spaCy first finds if a word is a NOUN, VERB, ADJ… because *saw* (tool) and *saw* (past of see) have different lemmas.
* **A lemma is not a synonym** – *children* and *kids* stay different (synonyms = Exercise 7).

## What the script does

The script `ex4_lemmatization.py` follows the exercise sheet step by step; every step is commented in the code.

| # | Task | How |
|---|---|---|
| 1 | Tokenize + lemmatize a sentence from the user | table TOKEN / LEMMA / POS / CHANGED? |
| 2 | Compare tokens vs lemmas | verbs → infinitive, plural nouns → singular |
| 3 | Filter stopwords and punctuation | content lemmas only |
| 4 | Challenge: unique lemmas of a paragraph | unique words vs unique lemmas |

## How to run

In a terminal (Command Prompt, PowerShell or the VS Code terminal):

```bash
cd "Chapter 1 - Introduction to NLP and Text Preprocessing/Exercise 4 - Lemmatization and Tokenization"
python ex4_lemmatization.py --sentence "The mice ran away."
```
You can also run it from any other folder: the script always finds its data and writes
into its own `outputs/` folder.  First time? Run `setup_environment.bat` in the main folder.

Needs: `pip install spacy` and `python -m spacy download en_core_web_sm`.

## Results

> **Graphs:** this exercise needs `spaCy`.  Run the script once on your computer (after `setup_environment.bat`): the graphs listed below will appear in the `outputs/` folder.

## Common mistakes

* Lowercasing **before** spaCy: the POS tagger uses capitals, so lowercase the *lemma* at the end.
* Importing another exercise file (`from count_tokens import token`): it runs that file's `input()` and crashes.

## Files

- `Exercise_4-Lemmatization and Tokenization.docx` – the exercise sheet (from Moodle)
- `ex4_lemmatization.py` – the solution, with comments
- `outputs/` – created by the script:
  - ex4_tokens_vs_lemmas.png – vocabulary size before/after lemmatization + top 10 lemmas
