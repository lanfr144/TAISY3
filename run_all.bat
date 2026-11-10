@echo off
REM ==========================================================================
REM  TAISY3 - run every exercise once and regenerate all the graphs
REM  (run setup_environment.bat first).  Each script writes into its own
REM  "outputs" folder.  The whole run takes about 10 minutes (IMDB is big).
REM ==========================================================================
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set C1=Chapter 1 - Introduction to NLP and Text Preprocessing
set C2=Chapter 2 - Traditional NLP
set C3=Chapter 3 - Neural Networks

call :run "%C1%\Exercise 1 - Regular Expressions\ex1_regular_expressions.py"
call :run "%C1%\Exercise 2 - Regular Expressions - Social Media\ex2_regex_social_media.py"
call :run "%C1%\Exercise 3 - Tokenization\ex3_tokenization.py" --sentence "The students aren't sure why the NLP course is so exciting!"
call :run "%C1%\Exercise 4 - Lemmatization and Tokenization\ex4_lemmatization.py" --sentence "The children were running faster than the geese."
call :run "%C1%\Exercise 5 - Bag of Words\ex5_bag_of_words.py"
call :run "%C1%\Exercise 6 -TF-IDF\ex6_tfidf.py"
call :run "%C1%\Exercise 7 - Thesaurus\ex7_thesaurus.py"
call :run "%C2%\Exercise 1 - Sentiment Analysis\ex1_sentiment_vader.py"
call :run "%C2%\Exercise 2 - Sentiment Analysis IMDB\ex2_sentiment_imdb_vader.py"
call :run "%C2%\Exercise 3 - Spam classification\ex3_spam_naive_bayes.py"
call :run "%C2%\Exercise 4 - Customer Reviews\ex4_reviews_naive_bayes.py"
call :run "%C2%\Exercise 5 - Topic Analysis\ex5_topic_analysis_nmf.py"
call :run "%C3%\Exercise 1 - Spam Classification\ex1_spam_mlp.py"
call :run "%C3%\Exercise 2 - IMDB Sentiment Analysis\ex2_imdb_mlp.py"

echo.
echo All exercises finished.  Open the "outputs" folders to see the graphs.
pause
exit /b 0

:run
echo.
echo ==========================================================================
echo  %~1
echo ==========================================================================
REM The output of each script is also saved next to it as run_output.txt
python %* > "%~dp1run_output.txt" 2>&1
if errorlevel 1 (
    echo   FAILED - see "%~dp1run_output.txt"
) else (
    echo   OK     - output in "%~dp1run_output.txt"
)
exit /b 0
