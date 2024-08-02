@echo off


IF NOT EXIST venv (
    python -m venv venv
    call %~dp0\venv\Scripts\activate.bat
) ELSE (
    echo venv folder already exists, skipping creation...
)
call %~dp0\venv\Scripts\activate.bat
set PYTHON="%~dp0\venv\Scripts\Python.exe"

echo venv %PYTHON%
%PYTHON% -m pip install -r cuda_requirements.txt
%PYTHON% -m pip install -r requirements.txt

%PYTHON% "%~dp0\download_models.py"


set /p VIDEO_PATH=Enter the path to your video file (e.g., C:\path\to\your\video\file.mp4):
set /p AUDIO_PATH=Enter the path to your audio file (e.g., C:\path\to\your\audio\file.wav):

echo Video Path: %VIDEO_PATH%
echo Audio Path: %AUDIO_PATH%

%PYTHON% inference.py --checkpoint_path "%~dp0\checkpoints\wav2lip_gan.pth" --face %VIDEO_PATH% --audio %AUDIO_PATH%

echo.
echo Launch unsuccessful. Exiting.
pause
