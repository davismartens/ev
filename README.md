
# Generate secret (OpenSSL in Bash)
`openssl rand -hex 32` (Bash) or `[Convert]::ToBase64String((1..24 | ForEach-Object { [byte](Get-Random -Maximum 256) }))` (powershell)


# Create venv
python -m venv .venv

## Delete venv
rm -r .venv

## install requirements
`uv pip install -r requirements.txt` (uv) or `pip install -r requirements.txt` (pip)


<!-- uvicorn app\playwright_runner\main:app --reload -->

# remove all __pycache__
Get-ChildItem -Path . -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
