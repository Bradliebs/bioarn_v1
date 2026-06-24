$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = "C:\pylibs;$projectRoot"
Set-Location $projectRoot

# Run all fast tests
python -m pytest tests\ -v --tb=short -m "not slow"

# Run with coverage
python -m pytest tests\ -m "not slow" --cov=bioarn --cov-report=html --cov-report=xml
