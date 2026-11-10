# Chapter 2 – Traditional NLP

[🏠 Main page](../README.md)

Three families of "classic" (non-neural) NLP methods:

| Family | Idea | Exercises |
|---|---|---|
| **Rule-based** | a human-made dictionary of scored words (VADER) – no training | 1, 2 |
| **Supervised learning** | learn from labelled examples (Multinomial Naive Bayes) | 3, 4 |
| **Unsupervised learning** | find hidden structure without labels (NMF topic modelling) | 5 |

## Exercises

| Exercise | Goal | Script |
|---|---|---|
| [Exercise 1 - Sentiment Analysis](Exercise%201%20-%20Sentiment%20Analysis/README.md) | Score 11 product reviews with VADER, label each one positive / negative / neutral, and save the reviews with their compound score and label in a new CSV. | `ex1_sentiment_vader.py` |
| [Exercise 2 - Sentiment Analysis IMDB](Exercise%202%20-%20Sentiment%20Analysis%20IMDB/README.md) | Run VADER on 50,000 IMDB movie reviews (after removing their HTML), measure its accuracy against the real labels, show the most positive/negative reviews, test several neutral thresholds and draw a pie chart. | `ex2_sentiment_imdb_vader.py` |
| [Exercise 3 - Spam classification](Exercise%203%20-%20Spam%20classification/README.md) | Train a Multinomial Naive Bayes classifier that detects spam e-mails, evaluate it on unseen e-mails and with cross-validation, and compare `CountVectorizer` with `TfidfVectorizer`. | `ex3_spam_naive_bayes.py` |
| [Exercise 4 - Customer Reviews](Exercise%204%20-%20Customer%20Reviews/README.md) | Train a Naive Bayes classifier that sorts customer reviews into three classes, so that the important complaints can be answered first. | `ex4_reviews_naive_bayes.py` |
| [Exercise 5 - Topic Analysis](Exercise%205%20-%20Topic%20Analysis/README.md) | Discover the hidden topics of ~18,000 newsgroup posts with TF-IDF + Non-negative Matrix Factorization, show the 10 top words of each topic and compare the topics with the real categories. | `ex5_topic_analysis_nmf.py` |

**Libraries used:** pandas, scikit-learn, NLTK (VADER), matplotlib

Each exercise folder has its own README with the key ideas, the tasks, how to run it, the results and the common mistakes.
