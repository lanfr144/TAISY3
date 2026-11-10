# Chapter 1 – Exercise 7: Thesaurus – Synonyms & Antonyms (WordNet)

[⬅ Back to the chapter](../README.md) · [🏠 Main page](../../README.md)

📄 **Exercise sheet:** [Exercise_7_Thesaurus_Synonyms_Antonyms.docx](Exercise_7_Thesaurus_Synonyms_Antonyms.docx)

## Goal

Use NLTK's WordNet as a thesaurus: find synonyms and antonyms, test whether two words share a meaning, and reflect on why a thesaurus without context can mislead.

## Key ideas (for a junior)

* **Synset** – one *meaning* of a word, e.g. `car.n.01` (word . part of speech . number).
* **Lemma** – one way of writing a meaning: `car.n.01` → car, auto, automobile, machine, motorcar.
* **Antonyms** are attached to **lemmas**, not to synsets: `lemma.antonyms()`.
* **Polysemy** – one word, many meanings (*bank*). Choosing the right one from the context = **word sense disambiguation** (`nltk.wsd.lesk` shows a simple version).

## What the script does

The script `ex7_thesaurus.py` follows the exercise sheet step by step; every step is commented in the code.

| # | Task | How |
|---|---|---|
| 1 | Import WordNet, download the corpus | `ensure_wordnet()` |
| 2 | `get_synonyms(word)` | synsets → lemmas → `set()` |
| 3 | Test: happy, fast, small, car |  |
| 4 | `get_antonyms(word)` – test hot, happy, fast, small | `lemma.antonyms()` |
| 5 | Readable output | `happy -> Synonyms: {...} | Antonyms: {...}` |
| 6 | Bonus: `same_meaning(w1, w2)` | intersection of synsets – car/automobile, buy/purchase, car/dog |
| 7 | Reflection: *bank* | all meanings + why it misleads + `lesk` demo |

## How to run

In a terminal (Command Prompt, PowerShell or the VS Code terminal):

```bash
cd "Chapter 1 - Introduction to NLP and Text Preprocessing/Exercise 7 - Thesaurus"
python ex7_thesaurus.py
```
You can also run it from any other folder: the script always finds its data and writes
into its own `outputs/` folder.  First time? Run `setup_environment.bat` in the main folder.

Needs: `pip install nltk` (WordNet is downloaded automatically the first time).

## Results

> **Graphs:** this exercise needs `NLTK`.  Run the script once on your computer (after `setup_environment.bat`): the graphs listed below will appear in the `outputs/` folder.

## Common mistakes

* Returning a list instead of a `set` → many duplicates.
* Looking for antonyms on the synset (`synset.antonyms()` does not exist).
* Forgetting that WordNet writes multi-word lemmas with `_` (`ice_cream`).

## Files

- `Exercise_7_Thesaurus_Synonyms_Antonyms.docx` – the exercise sheet (from Moodle)
- `ex7_thesaurus.py` – the solution, with comments
- `outputs/` – created by the script:
  - ex7_thesaurus_overview.png – meanings / synonyms / antonyms per word, meanings of *bank* per part of speech
