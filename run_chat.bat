@echo off
title Local Chat with Memory
rem Always run from the folder that holds this .bat so chats.db lands next to it
cd /d "%~dp0"

if not exist "%~dp0chat.py" (
    echo Could not find chat.py next to this .bat file.
    pause
    exit /b 1
)

rem Start the Ollama server only if it is not already answering
curl -s -o nul http://localhost:11434
if errorlevel 1 (
    echo Starting Ollama...
    start "" /min ollama serve
    timeout /t 4 /nobreak >nul
)

rem Pass a model name to switch models, e.g.  run_chat.bat llama3.1:8b
python "%~dp0chat.py" %*

echo.
pause
