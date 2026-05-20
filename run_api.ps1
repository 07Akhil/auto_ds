$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -m uvicorn autods_gpt.api:app --host 127.0.0.1 --port 8000
