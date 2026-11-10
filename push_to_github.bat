@echo off
REM ==========================================================================
REM  TAISY3 - publish this EXERCISES folder to https://github.com/lanfr144/TAISY3
REM
REM  Needs Git for Windows (https://git-scm.com).  The first push opens a
REM  browser window so you can sign in to GitHub (Git Credential Manager).
REM  What is NOT sent: see .gitignore (exclude_from_git, the .docx/.pptx course
REM  files, the uncompressed 66 MB IMDB csv).
REM ==========================================================================
setlocal
cd /d "%~dp0"
set REPO=https://github.com/lanfr144/TAISY3.git

where git >nul 2>nul || (echo Git is not installed: https://git-scm.com/download/win & pause & exit /b 1)

REM --- who signs the commits? (asked only once) ----------------------------
for /f "delims=" %%i in ('git config user.name') do set GN=%%i
if not defined GN (
    set /p GN="Your name for the commits (e.g. lanfr144): "
    call git config --global user.name "%%GN%%"
)
for /f "delims=" %%i in ('git config user.email') do set GE=%%i
if not defined GE (
    set /p GE="Your e-mail for the commits (it becomes visible on GitHub): "
    call git config --global user.email "%%GE%%"
)

REM --- first time only: connect this folder to the GitHub repository --------
REM The repository already has a README and a LICENSE ("Initial commit").
REM We attach our files ON TOP of that commit, so nothing on GitHub is lost.
if not exist ".git" (
    echo Connecting this folder to %REPO% ...
    git init -b main || goto :error
    git remote add origin %REPO% || goto :error
    git fetch origin || goto :error
    git reset --mixed origin/main || goto :error
)

echo.
echo Files that will be sent:
git add -A || goto :error
git status --short
echo.
set /p OK="Commit and push these changes? (y/n) "
if /i not "%OK%"=="y" (echo Cancelled. & pause & exit /b 0)

git commit -m "TAISY3 exercises: commented solutions, graphs and README files" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01C1eqtmU39ZsnPAWBSdasSc"
git push -u origin main || goto :error

echo.
echo Done!  See https://github.com/lanfr144/TAISY3
pause
exit /b 0

:error
echo.
echo Something went wrong (see the message above).  Nothing was deleted.
pause
exit /b 1
