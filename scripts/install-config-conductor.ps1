# Software install for the twin and the instrument PC. Starts nothing.
# Installs: uv (if missing, per-user, no administrator rights), Python 3.12
# (version from .python-version) and the packages pinned in uv.lock, into .venv.

param([switch]$DryRun)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)

function Find-Uv {
    $c = Get-Command uv -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    # The Astral installer puts uv here; this process's PATH predates it.
    $p = Join-Path $env:USERPROFILE '.local\bin\uv.exe'
    if (Test-Path $p) { return $p }
    return $null
}

$uv = Find-Uv
Write-Host ''
Write-Host 'This will install, on this computer:' -ForegroundColor Cyan
if ($uv) { Write-Host "  - uv: already installed ($uv)" } else { Write-Host '  - uv (the Python tool manager), from astral.sh, for your user only' }
Write-Host '  - Python 3.12, if uv does not already have it'
Write-Host '  - the program''s packages, into the .venv folder here (from uv.lock)'
Write-Host 'Nothing is started.'
if ($DryRun) { Write-Host '[dry run] nothing installed.'; exit 0 }
if ((Read-Host 'Install now? [y/N]') -notmatch '^(y|yes)$') { Write-Host 'Nothing installed.'; exit 0 }

if (-not $uv) {
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $uv = Find-Uv
    if (-not $uv) { Write-Host 'uv installed but not found. Close this window and run again.' -ForegroundColor Red; exit 1 }
}
& $uv sync
if ($LASTEXITCODE -ne 0) { Write-Host 'uv sync failed (see above).' -ForegroundColor Red; exit 1 }
Write-Host ''
Write-Host 'Done. Run the digital twin with run-digital-twin.cmd.' -ForegroundColor Green
Write-Host 'On an instrument PC, now run install-config-hardware.cmd.'
