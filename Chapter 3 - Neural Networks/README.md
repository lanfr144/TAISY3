# Chapter 3 – Neural Networks

[🏠 Main page](../README.md)

The same problems as in Chapter 2, solved with a **Multi-Layer Perceptron** (scikit-learn's `MLPClassifier`),
so you can compare a neural network with Naive Bayes on a tiny dataset (48 e-mails) and on a big one (50,000 reviews).

**The lesson:** on 48 e-mails the network brings nothing; on 50,000 reviews it beats Naive Bayes (89.9 % vs 86.3 %).
Neural networks need data.

## Exercises

| Exercise | Goal | Script |
|---|---|---|
| [Exercise 1 - Spam Classification](Exercise%201%20-%20Spam%20Classification/README.md) | Solve the spam problem of Chapter 2 again, this time with a Multi-Layer Perceptron, compare CountVectorizer with TF-IDF, and compare the network with Naive Bayes. | `ex1_spam_mlp.py` |
| [Exercise 2 - IMDB Sentiment Analysis](Exercise%202%20-%20IMDB%20Sentiment%20Analysis/README.md) | Clean the 50,000 IMDB reviews, turn them into a TF-IDF matrix, classify them with MLPs using different parameters, and add a *neutral* class from the network's confidence. | `ex2_imdb_mlp.py` |

**Libraries used:** pandas, scikit-learn, matplotlib

Each exercise folder has its own README with the key ideas, the tasks, how to run it, the results and the common mistakes.
