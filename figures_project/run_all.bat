@echo off
chcp 65001 > nul
cd /d "%~dp0"
python -m pip install -r requirements.txt
python plot_all.py --project support --output publication_figures --dpi 300
if errorlevel 1 (
    echo Figure generation failed. Please inspect output messages.
    exit /b 1
)
echo All figures generated successfully.
pause
