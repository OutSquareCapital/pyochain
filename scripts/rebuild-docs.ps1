# Rebuild documentation with complete cache clearing
# Usage: .\rebuild-docs.ps1
uv run scripts/generate_docs.py
Write-Host "Clearing cache..." -ForegroundColor Yellow
Remove-Item -Recurse -Force site/ -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .cache/ -ErrorAction SilentlyContinue

Write-Host "Building documentation..." -ForegroundColor Cyan
uv run zensical build

Write-Host "Resolving cross-references..." -ForegroundColor Cyan
uv run scripts/fix_autorefs.py

uv run zensical serve