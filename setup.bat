@echo off
python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
echo Setup complete! Select .venv in VS Code.
pause