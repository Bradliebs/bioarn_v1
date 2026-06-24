$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = "C:\pylibs;$projectRoot"
Set-Location $projectRoot

# Basic code quality checks
python -m compileall bioarn tests
python -m ruff check bioarn tests --select E9,F63,F7,F82
python -m pytest tests\ --co -q
