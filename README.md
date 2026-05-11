# AI Stock Market Prediction System

This is a cleaned local version of the Colab stock prediction project.
It includes login, signup, a user dashboard, watchlist saving, and the stock prediction dashboard.

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The app expects `nifty500_stocks.csv` to be in the same folder as `app.py`.
User accounts are stored locally in `users.json` after signup. Passwords are stored with a salted hash, not plain text.

On this Windows machine, Python was installed at:

```text
C:\Users\YASH BANSAL\AppData\Local\Programs\Python\Python312\python.exe
```

You can also double-click `run_app.bat`, or run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_app.ps1
```

## Deploy

This project is ready for Streamlit Community Cloud.

1. Push this folder to GitHub.
2. Open https://share.streamlit.io/
3. Click "New app".
4. Select the GitHub repository.
5. Set the main file path to `app.py`.
6. Deploy.

Note: Streamlit Community Cloud storage can reset when the app restarts. For production user accounts, connect a real database instead of local `users.json`.
