# Shared data

[🏠 Main page](../README.md)

| File | Used by | Notes |
|---|---|---|
| `Ch2_Ex1_IMDB.csv.zip` | Chapter 2 – Exercise 2, Chapter 3 – Exercise 2 | 50,000 IMDB movie reviews (`review`, `sentiment` = positive / negative, 25,000 each). Compressed to 26 MB so it fits on GitHub. |
| `Ch2_Ex1_IMDB.csv` | same | The uncompressed version (66 MB). **Not on GitHub** – the scripts use the `.zip` automatically when the `.csv` is missing. |

`pandas.read_csv("Ch2_Ex1_IMDB.csv.zip")` reads the zip directly, no need to unzip it.

The small datasets (spam e-mails, customer reviews, Alice passages) are stored directly in the exercise folder that uses them, as on Moodle.
