# run.ps1
$ErrorActionPreference = "Stop"

Write-Host "Starting FastAPI backend..."

# aktifkan virtual environment
if (-Not (Test-Path ".\venv\Scripts\Activate.ps1")) {
    Write-Host "venv not found, please run: python -m venv venv"
    exit
}
.\venv\Scripts\Activate.ps1

# set lokasi artifacts
$env:ART_DIR = ".\artifacts"

# jalankan server
uvicorn src.serve_tfidf:app --reload --port 8000
