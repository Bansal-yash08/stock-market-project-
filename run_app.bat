@echo off
cd /d "%~dp0"
"C:\Users\YASH BANSAL\AppData\Local\Programs\Python\Python312\python.exe" -m pip install -r requirements.txt
"C:\Users\YASH BANSAL\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run app.py
pause
