# Chapter 1 – Introduction to NLP and Text Preprocessing

[🏠 Main page](../README.md)

Before a computer can learn anything from text, the text must be **cleaned** and turned into **numbers**.
This chapter covers the full preprocessing toolbox:

```
raw text ─► regex cleaning ─► tokens ─► stopwords / lemmas ─► Bag of Words / TF-IDF ─► numbers for a model
           (Ex. 1-2)         (Ex. 3)    (Ex. 3-4)              (Ex. 5-6)
```
Exercise 7 adds *meaning*: synonyms and antonyms with the WordNet thesaurus.

## Exercises

| Exercise | Goal | Script |
|---|---|---|
| [Exercise 1 - Regular Expressions](Exercise%201%20-%20Regular%20Expressions/README.md) | Clean a small dataset of 15 sentences and extract information from it (e-mails, URLs, phone numbers, dates…) with regular expressions and the pandas `.str` methods. | `ex1_regular_expressions.py` |
| [Exercise 2 - Regular Expressions - Social Media](Exercise%202%20-%20Regular%20Expressions%20-%20Social%20Media/README.md) | Turn 8 messy social-media posts (mentions, hashtags, emojis, slang, `happppyyyy`…) into clean text ready for NLP, using placeholders such as `<URL>`, `<EMOJI>`, `<NUM>`. | `ex2_regex_social_media.py` |
| [Exercise 3 - Tokenization](Exercise%203%20-%20Tokenization/README.md) | Ask the user for a sentence, cut it into tokens with spaCy, count them, remove punctuation and stopwords, lowercase them, and list the unique words of a paragraph. | `ex3_tokenization.py` |
| [Exercise 4 - Lemmatization and Tokenization](Exercise%204%20-%20Lemmatization%20and%20Tokenization/README.md) | Show every token next to its lemma, compare what changes for verbs and plural nouns, keep only the content lemmas, and list the unique lemmas of a paragraph. | `ex4_lemmatization.py` |
| [Exercise 5 - Bag of Words](Exercise%205%20-%20Bag%20of%20Words/README.md) | Read three passages of *Alice's Adventures in Wonderland*, build a Document-Term Matrix with `CountVectorizer`, show it as a DataFrame and find the most frequent words, with and without stopwords. | `ex5_bag_of_words.py` |
| [Exercise 6 -TF-IDF](Exercise%206%20-TF-IDF/README.md) | Analyse the same three Alice passages with `TfidfVectorizer`, find the 5 most characteristic words of each document and compare with Bag of Words. | `ex6_tfidf.py` |
| [Exercise 7 - Thesaurus](Exercise%207%20-%20Thesaurus/README.md) | Use NLTK's WordNet as a thesaurus: find synonyms and antonyms, test whether two words share a meaning, and reflect on why a thesaurus without context can mislead. | `ex7_thesaurus.py` |

**Libraries used:** pandas, scikit-learn, spaCy (`en_core_web_sm`), NLTK (WordNet)

Each exercise folder has its own README with the key ideas, the tasks, how to run it, the results and the common mistakes.
