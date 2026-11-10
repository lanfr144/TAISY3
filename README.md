# TAISY3

Cours de Taisy3 du Lycée des Arts et Métiers de Luxembourg – **solved exercises** with comments, graphs and explanations written for junior students.

## Quick start

1. Install [Python 3.10+](https://www.python.org/downloads/) (tick *Add Python to PATH*).
2. Double-click **`setup_environment.bat`** – installs the packages of `requirements.txt` and downloads the spaCy model and the NLTK data.
3. Run an exercise, for example:
   ```bash
   cd "Chapter 1 - Introduction to NLP and Text Preprocessing/Exercise 5 - Bag of Words"
   python ex5_bag_of_words.py
   ```
   or double-click **`run_all.bat`** to run all 14 exercises (about 10 minutes).

Every script:
- starts with a long comment that explains the key ideas,
- prints each task of the exercise sheet clearly in the terminal,
- saves its graphs and result files in its own `outputs/` folder,
- works from any folder (it finds its data next to itself).

## Exercises

### [Chapter 1 – Introduction to NLP and Text Preprocessing](Chapter%201%20-%20Introduction%20to%20NLP%20and%20Text%20Preprocessing/README.md)

| Exercise | Goal |
|---|---|
| [Exercise 1 - Regular Expressions](Chapter%201%20-%20Introduction%20to%20NLP%20and%20Text%20Preprocessing/Exercise%201%20-%20Regular%20Expressions/README.md) | Clean a small dataset of 15 sentences and extract information from it (e-mails, URLs, phone numbers, dates…) with regular expressions and the pandas `.str` methods. |
| [Exercise 2 - Regular Expressions - Social Media](Chapter%201%20-%20Introduction%20to%20NLP%20and%20Text%20Preprocessing/Exercise%202%20-%20Regular%20Expressions%20-%20Social%20Media/README.md) | Turn 8 messy social-media posts (mentions, hashtags, emojis, slang, `happppyyyy`…) into clean text ready for NLP, using placeholders such as `<URL>`, `<EMOJI>`, `<NUM>`. |
| [Exercise 3 - Tokenization](Chapter%201%20-%20Introduction%20to%20NLP%20and%20Text%20Preprocessing/Exercise%203%20-%20Tokenization/README.md) | Ask the user for a sentence, cut it into tokens with spaCy, count them, remove punctuation and stopwords, lowercase them, and list the unique words of a paragraph. |
| [Exercise 4 - Lemmatization and Tokenization](Chapter%201%20-%20Introduction%20to%20NLP%20and%20Text%20Preprocessing/Exercise%204%20-%20Lemmatization%20and%20Tokenization/README.md) | Show every token next to its lemma, compare what changes for verbs and plural nouns, keep only the content lemmas, and list the unique lemmas of a paragraph. |
| [Exercise 5 - Bag of Words](Chapter%201%20-%20Introduction%20to%20NLP%20and%20Text%20Preprocessing/Exercise%205%20-%20Bag%20of%20Words/README.md) | Read three passages of *Alice's Adventures in Wonderland*, build a Document-Term Matrix with `CountVectorizer`, show it as a DataFrame and find the most frequent words, with and without stopwords. |
| [Exercise 6 -TF-IDF](Chapter%201%20-%20Introduction%20to%20NLP%20and%20Text%20Preprocessing/Exercise%206%20-TF-IDF/README.md) | Analyse the same three Alice passages with `TfidfVectorizer`, find the 5 most characteristic words of each document and compare with Bag of Words. |
| [Exercise 7 - Thesaurus](Chapter%201%20-%20Introduction%20to%20NLP%20and%20Text%20Preprocessing/Exercise%207%20-%20Thesaurus/README.md) | Use NLTK's WordNet as a thesaurus: find synonyms and antonyms, test whether two words share a meaning, and reflect on why a thesaurus without context can mislead. |

### [Chapter 2 – Traditional NLP](Chapter%202%20-%20Traditional%20NLP/README.md)

| Exercise | Goal |
|---|---|
| [Exercise 1 - Sentiment Analysis](Chapter%202%20-%20Traditional%20NLP/Exercise%201%20-%20Sentiment%20Analysis/README.md) | Score 11 product reviews with VADER, label each one positive / negative / neutral, and save the reviews with their compound score and label in a new CSV. |
| [Exercise 2 - Sentiment Analysis IMDB](Chapter%202%20-%20Traditional%20NLP/Exercise%202%20-%20Sentiment%20Analysis%20IMDB/README.md) | Run VADER on 50,000 IMDB movie reviews (after removing their HTML), measure its accuracy against the real labels, show the most positive/negative reviews, test several neutral thresholds and draw a pie chart. |
| [Exercise 3 - Spam classification](Chapter%202%20-%20Traditional%20NLP/Exercise%203%20-%20Spam%20classification/README.md) | Train a Multinomial Naive Bayes classifier that detects spam e-mails, evaluate it on unseen e-mails and with cross-validation, and compare `CountVectorizer` with `TfidfVectorizer`. |
| [Exercise 4 - Customer Reviews](Chapter%202%20-%20Traditional%20NLP/Exercise%204%20-%20Customer%20Reviews/README.md) | Train a Naive Bayes classifier that sorts customer reviews into three classes, so that the important complaints can be answered first. |
| [Exercise 5 - Topic Analysis](Chapter%202%20-%20Traditional%20NLP/Exercise%205%20-%20Topic%20Analysis/README.md) | Discover the hidden topics of ~18,000 newsgroup posts with TF-IDF + Non-negative Matrix Factorization, show the 10 top words of each topic and compare the topics with the real categories. |

### [Chapter 3 – Neural Networks](Chapter%203%20-%20Neural%20Networks/README.md)

| Exercise | Goal |
|---|---|
| [Exercise 1 - Spam Classification](Chapter%203%20-%20Neural%20Networks/Exercise%201%20-%20Spam%20Classification/README.md) | Solve the spam problem of Chapter 2 again, this time with a Multi-Layer Perceptron, compare CountVectorizer with TF-IDF, and compare the network with Naive Bayes. |
| [Exercise 2 - IMDB Sentiment Analysis](Chapter%203%20-%20Neural%20Networks/Exercise%202%20-%20IMDB%20Sentiment%20Analysis/README.md) | Clean the 50,000 IMDB reviews, turn them into a TF-IDF matrix, classify them with MLPs using different parameters, and add a *neutral* class from the network's confidence. |

### [Chapter 4 – Convolutional Neural Networks](Chapter%204%20-%20Convolutional%20Neural%20Networks/README.md)

Not published yet.

## Repository structure

```
EXERCISES/
├── README.md                  ← this page
├── requirements.txt           ← Python packages
├── setup_environment.bat      ← install everything (Windows)
├── run_all.bat                ← run all exercises
├── push_to_github.bat         ← publish to github.com/lanfr144/TAISY3
├── data/                      ← the shared IMDB dataset
├── Chapter 1 - .../Exercise N - .../
│   ├── README.md              ← explanations + results
│   ├── Exercise_N-....docx    ← the exercise sheet
│   ├── exN_....py             ← the solution
│   ├── <small dataset>        ← if the exercise has one
│   └── outputs/               ← graphs and CSV files created by the script
├── Chapter 2 - ... / Chapter 3 - ... / Chapter 4 - ...
└── exclude_from_git/          ← old versions, kept locally only (see .gitignore)
```

Each exercise folder contains its exercise sheet (`.docx`, from Moodle) next to the solution.

## About the course

**BTS Applied Artificial Intelligence** — Course *Training of AI Systems 3*

### A. Organisation

| | |
|---|---|
| Code | TAISY3 |
| In-person training | 60h |
| Type | Compulsory |
| Instruction language | English |
| Mode of delivery | Face-to-face |
| Continuous assessment | No |
| Exam | Yes |
| Number of assessments | 1 |
| Autonomous work | 40h |
| ECTS | 4 |

### B. Course prerequisites

In order to be allowed to participate in this course, a student must meet the following prerequisites:

- Have obtained a grade of at least 10 in the course Training of AI Systems 1 or have a validation of equivalent learning outcomes.
- Have followed the course Training of AI Systems 2 or have a validation of equivalent learning outcomes.

### C. Learning outcomes

On successful completion of this course, students will be able to:

- Understand the predictions issued by a model,
- Describe and apply natural language processing techniques using pre-existing implementations,
- Describe and apply advanced learning and training approaches using pre-existing implementations,
- Understand the application domain and required working conditions for natural language processing solutions.

### D. Course contents

The course will cover the following topics:

- Natural Language Processing
  - Text processing
  - Syntactics
  - Semantics
  - Discourse
  - …
- Advanced learning and training approaches
  - Applied examples and use of Neural Networks and Deep Learning methods.

> Example methods: Convolutional Neural Networks, Recurrent Neural Networks, Multi-Layer Perceptrons, Markov Models, ...

### E. Learning and teaching methods

The teacher should not go into details about the exact internal functioning and implementation of NLP as well as the advanced learning and training approaches. The focus should be on the use of already existing libraries and frameworks of the methods and how to use them in a given scenario. The student should be able to choose the fitting algorithm to solve the problem. Furthermore, Neural Networks are only to be introduced with a practical example and existing libraries. Students should know what Neural Networks are and the concept they are based on to complete their knowledge of the general subject, its applicability, and its limitations.

Learning and teaching methods will vary depending on the subject and may include:

- Teacher-centred
- Blended learning
- Flipped classroom
- Demonstration-based learning
- Inquiry-based learning

### F. Assessment methods

The students will be assessed in continuous assessment that can take various forms, depending on the final objectives. Those methods may involve:

- Practical test
- Written test
- Oral test

### G. Required readings

The following references are required readings:

- none

### H. Optional readings

The following references are optional readings:

- none

### I. Targeted certification

This module prepares the students do the following industrial certifications:

- none

