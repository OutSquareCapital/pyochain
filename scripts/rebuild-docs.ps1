# Rebuild documentation with complete cache clearing
# Usage: .\rebuild-docs.ps1

Write-Host "Clearing cache..." -ForegroundColor Yellow
Remove-Item -Recurse -Force site/ -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .cache/ -ErrorAction SilentlyContinue

Write-Host "Building documentation..." -ForegroundColor Cyan
uv run zensical serve