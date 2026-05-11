$ErrorActionPreference = "Stop"

$projectPython = "C:\Users\YASH BANSAL\AppData\Local\Programs\Python\Python312\python.exe"

if (Test-Path $projectPython) {
    $pythonPath = $projectPython
} else {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        $pythonPath = $python.Source
    }
}

if (-not $pythonPath) {
    Write-Host "Python is not installed or not on PATH."
    Write-Host "Install Python 3.10+ from https://www.python.org/downloads/windows/"
    Write-Host "During install, tick: Add python.exe to PATH"
    exit 1
}

& $pythonPath -m pip install -r requirements.txt
& $pythonPath -m streamlit run app.py
