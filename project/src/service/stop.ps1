$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    & $VenvPython (Join-Path $ProjectRoot "scripts\stop_services.py") @args
} else {
    py (Join-Path $ProjectRoot "scripts\stop_services.py") @args
}
