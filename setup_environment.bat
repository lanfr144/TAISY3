@echo off
REM ==========================================================================
REM  TAISY3 - install everything the exercises need (run once)
REM  Double-click this file, or run it from a terminal in this folder.
REM ==========================================================================
cd /d "%~dp0"

echo [1/3] Installing the Python packages...
python -m pip install -r requirements.txt || goto :error

echo [2/3] Downloading the spaCy English model (Chapter 1, exercises 3 and 4)...
python -m spacy download en_core_web_sm || goto :error

echo [3/3] Downloading the NLTK data (WordNet for exercise 7, VADER for chapter 2)...
python -c "import nltk; [nltk.download(p) for p in ('wordnet', 'omw-1.4', 'vader_lexicon')]" || goto :error

echo.
echo Done!  You can now run the exercises, or run_all.bat to run them all.
pause
exit /b 0

:error
echo.
echo Something went wrong.  Check that Python is installed and on your PATH
echo (in a terminal, "python --version" must work).
pause
exit /b 1
