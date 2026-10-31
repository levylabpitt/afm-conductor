<#
  install-config-hardware.ps1 -- ONE-TIME setup of an Asylum instrument PC.

  Double-click "install-config-hardware.cmd" at the repo root, or:

      powershell -ExecutionPolicy Bypass -File scripts\install-config-hardware.ps1
      powershell -ExecutionPolicy Bypass -File scripts\install-config-hardware.ps1 -DryRun
      powershell -ExecutionPolicy Bypass -File scripts\install-config-hardware.ps1 -UndoIgor

  Run install-config-conductor.cmd first (it installs uv and Python).
  Steps, each announced and confirmed with y/N (nothing installs until you agree):
    1. add the hardware packages to the environment, including pywin32 (COM to Igor)
    2. Igor 6.37 swap: Asylum ships Igor 6.38, which has no COM interface; only
       Igor.exe is replaced, the original is kept as Igor.exe.6.38-backup
    3. SOCKIT XOP (Igor <-> Python TCP transport)
    4. the program's Igor procedures
  -DryRun finds and reports everything but downloads, copies and installs nothing.
  -UndoIgor puts the original Igor.exe back. This script never starts the program.
  Manual version of every step: INSTRUMENT_SETUP.md.
#>

param([switch]$DryRun, [switch]$UndoIgor)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $repo

$Igor637Url = 'https://www.wavemetrics.net/Downloads/Win/setupIgor6.exe'
$BackupName = 'Igor.exe.6.38-backup'

function Say($text, $color = 'Gray') { Write-Host $text -ForegroundColor $color }

function Step($title) {
    Say ''
    Say "== $title" 'Cyan'
}

function Confirm-Step($question) {
    if ($DryRun) { Say "  [dry run] would ask: $question" 'DarkGray'; return $false }
    $a = Read-Host "  $question [y/N]"
    return ($a -match '^(y|yes)$')
}

function Find-Uv {
    $c = Get-Command uv -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    # The Astral installer puts uv here; this process's PATH predates it.
    $p = Join-Path $env:USERPROFILE '.local\bin\uv.exe'
    if (Test-Path $p) { return $p }
    return $null
}

function Assert-IgorClosed {
    $running = Get-Process -Name 'Igor*' -ErrorAction SilentlyContinue
    if ($running) {
        Say '  Igor is running. Close Igor completely (check the system tray), then run this again.' 'Red'
        exit 1
    }
}

function Get-IgorVersion($exe) {
    $v = (Get-Item -LiteralPath $exe).VersionInfo
    return "$($v.FileVersion)"
}

# WaveMetrics writes the version as 6.37 or 6.3.7 depending on the build.
function Test-Is637($exe) {
    $v = (Get-Item -LiteralPath $exe).VersionInfo
    return (("$($v.FileVersion) $($v.ProductVersion)") -match '6[.,]\s?(37|3[.,]\s?7)\b')
}

function Read-IgorPath($prompt) {
    $typed = (Read-Host "  $prompt").Trim().Trim('"')
    if (-not $typed) { return $null }
    if (Test-Path -LiteralPath $typed -PathType Container) { $typed = Join-Path $typed 'Igor.exe' }
    if (Test-Path -LiteralPath $typed -PathType Leaf) { return $typed }
    Say "  Not found: $typed" 'Yellow'
    return $null
}

# Asylum keeps Igor at <drive>\AsylumResearch\v*\<RealTime|Offline>\Igor Pro Folder\Igor.exe.
function Find-AsylumIgors {
    $roots = @('C:\AsylumResearch', 'D:\AsylumResearch')
    foreach ($d in (Get-PSDrive -PSProvider FileSystem)) {
        $roots += (Join-Path $d.Root 'AsylumResearch')
    }
    $found = @()
    foreach ($r in ($roots | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $r)) { continue }
        $found += Get-ChildItem -Path (Join-Path $r 'v*\*\Igor Pro Folder\Igor.exe') -ErrorAction SilentlyContinue
        $found += Get-ChildItem -Path (Join-Path $r '*\Igor Pro Folder\Igor.exe') -ErrorAction SilentlyContinue
    }
    return @($found | ForEach-Object { $_.FullName } | Select-Object -Unique)
}

# Returns the confirmed Asylum Igor.exe path, or $null (dry run never asks).
function Select-AsylumIgor {
    $cands = @(Find-AsylumIgors)
    if ($cands.Count -eq 0) {
        Say '  No Asylum Igor found under any AsylumResearch folder.' 'Yellow'
        if ($DryRun) { return $null }
        return (Read-IgorPath 'Type the full path to the Asylum Igor.exe (or its Igor Pro Folder), or Enter to skip')
    }
    Say '  Asylum Igor installs found:'
    for ($i = 0; $i -lt $cands.Count; $i++) {
        Say ("    [{0}] {1}   (version {2})" -f ($i + 1), $cands[$i], (Get-IgorVersion $cands[$i]))
    }
    if ($DryRun) { return $cands[0] }
    $a = (Read-Host "  Number to use, a different path, or Enter for [1]").Trim().Trim('"')
    if (-not $a) { return $cands[0] }
    if ($a -match '^\d+$' -and [int]$a -ge 1 -and [int]$a -le $cands.Count) { return $cands[[int]$a - 1] }
    if (Test-Path -LiteralPath $a -PathType Container) { $a = Join-Path $a 'Igor.exe' }
    if (Test-Path -LiteralPath $a -PathType Leaf) { return $a }
    Say "  Not found: $a" 'Red'
    return $null
}

function Test-FolderWritable($folder) {
    $t = Join-Path $folder ([IO.Path]::GetRandomFileName())
    try { New-Item -ItemType File -Path $t -ErrorAction Stop | Out-Null; Remove-Item -LiteralPath $t; return $true }
    catch { return $false }
}

# Runs a small script, elevated (one UAC prompt) only when the folder needs it.
function Invoke-FileOp($scriptText, $folder) {
    $tmp = Join-Path $env:TEMP 'afm-igor-swap.ps1'
    Set-Content -LiteralPath $tmp -Value $scriptText -Encoding UTF8
    $args2 = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$tmp`"")
    if (Test-FolderWritable $folder) {
        & powershell.exe @args2
    } else {
        Say '  That folder needs administrator rights. Approve the Windows prompt.' 'Yellow'
        Start-Process powershell.exe -Verb RunAs -Wait -ArgumentList $args2
    }
    Remove-Item -LiteralPath $tmp -ErrorAction SilentlyContinue
}

function Find-Igor637 {
    $paths = @()
    foreach ($pf in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if ($pf) { $paths += (Join-Path $pf 'WaveMetrics\Igor Pro Folder\Igor.exe') }
    }
    foreach ($p in $paths) {
        if ((Test-Path -LiteralPath $p) -and (Test-Is637 $p)) { return $p }
    }
    foreach ($pf in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if (-not $pf) { continue }
        $hit = Get-ChildItem -Path (Join-Path $pf 'WaveMetrics\*\Igor.exe') -ErrorAction SilentlyContinue |
            Where-Object { Test-Is637 $_.FullName } | Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    return $null
}

# ---------------------------------------------------------------------------
Say '======================================================================' 'Cyan'
Say '  AFM Conductor -- instrument PC setup (run once)' 'Cyan'
Say '======================================================================' 'Cyan'
Say "  folder: $repo"
if ($DryRun) { Say '  DRY RUN: reporting only; nothing is downloaded, copied or installed.' 'Yellow' }

# ---------------------------------------------------------------------------
if ($UndoIgor) {
    Step 'Restore the Asylum Igor.exe (Igor 6.38)'
    Assert-IgorClosed
    $asy = Select-AsylumIgor
    if (-not $asy) { Say '  No Igor.exe chosen; nothing changed.'; exit 1 }
    $dir = Split-Path $asy -Parent
    $bak = Join-Path $dir $BackupName
    if (-not (Test-Path -LiteralPath $bak)) { Say "  No backup found at $bak; nothing to restore." 'Red'; exit 1 }
    Say "  Backup: $bak (version $(Get-IgorVersion $bak))"
    if (-not (Confirm-Step 'Restore it over Igor.exe?')) { Say '  Nothing changed.'; exit 0 }
    Invoke-FileOp "Copy-Item -LiteralPath '$bak' -Destination '$asy' -Force" $dir
    Say "  Igor.exe is now version $(Get-IgorVersion $asy)" 'Green'
    exit 0
}

# --- 1. hardware packages ---------------------------------------------------
Step '1/4  Hardware packages (uv sync, includes pywin32 for COM)'
$uv = Find-Uv
if (-not $uv -or -not (Test-Path (Join-Path $repo '.venv\Scripts\python.exe'))) {
    Say '  Run install-config-conductor.cmd first.' 'Red'
    exit 1
}
if (Confirm-Step 'Add the hardware packages now?') {
    & $uv sync --extra hardware --extra geometry
    if ($LASTEXITCODE -ne 0) { Say '  uv sync failed (see above).' 'Red'; exit 1 }
    Say '  Environment ready.' 'Green'
} else { Say '  Skipped.' }

# --- 2. Igor 6.37 swap ------------------------------------------------------
Step '2/4  Igor 6.37 (Asylum ships 6.38, which has no COM interface)'
Assert-IgorClosed
$asy = Select-AsylumIgor
if (-not $asy) {
    Say '  No Asylum Igor chosen. Steps 2-4 need it; run this again once it is found.' 'Yellow'
    if ($DryRun) { Say ''; Say 'Dry run finished.' 'Green' }
    exit $(if ($DryRun) { 0 } else { 1 })
}
$asyDir = Split-Path $asy -Parent
Say "  Asylum Igor: $asy  (version $(Get-IgorVersion $asy))"

if (Test-Is637 $asy) {
    Say '  Already Igor 6.37. Nothing to swap.' 'Green'
} elseif (Confirm-Step 'Swap the Asylum Igor.exe for Igor 6.37?') {
    $new = Find-Igor637
    if ($new -and (Confirm-Step "Igor 6.37 is already installed at $new. Use it instead of downloading?")) {
        Say "  Using $new"
    } else {
        $new = $null
        $dlDir = Join-Path $env:TEMP ('afm-igor637-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
        New-Item -ItemType Directory -Path $dlDir | Out-Null
        $dl = Join-Path $dlDir 'setupIgor6.exe'
        Say "  Downloading $Igor637Url ..."
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $Igor637Url -OutFile $dl -UseBasicParsing
        Say '  Running the WaveMetrics installer. Approve the Windows prompt and click through with the defaults.' 'Yellow'
        Say '  (The existing Asylum licence carries over; nothing to enter.)'
        Start-Process -FilePath $dl -Verb RunAs -Wait
        Remove-Item -LiteralPath $dlDir -Recurse -ErrorAction SilentlyContinue
        $new = Find-Igor637
    }
    while (-not $new -or -not (Test-Is637 $new)) {
        if ($new) { Say "  That Igor.exe reports version $(Get-IgorVersion $new), not 6.37." 'Yellow' }
        $new = Read-IgorPath 'Type the full path to the new 6.37 Igor.exe, or Enter to stop'
        if (-not $new) { Say '  Stopped. The Asylum Igor.exe was not changed.' 'Red'; exit 1 }
    }
    Say "  Igor 6.37 found: $new"

    Assert-IgorClosed
    $bak = Join-Path $asyDir $BackupName
    $ops = @('$ErrorActionPreference = "Stop"')
    if (Test-Path -LiteralPath $bak) {
        Say "  A backup already exists ($bak); keeping it untouched."
    } else {
        $ops += "Rename-Item -LiteralPath '$asy' -NewName '$BackupName'"
        $ops += "Copy-Item -LiteralPath '$new' -Destination '$asy'"
    }
    if (Test-Path -LiteralPath $bak) { $ops += "Copy-Item -LiteralPath '$new' -Destination '$asy' -Force" }
    Invoke-FileOp ($ops -join "`n") $asyDir
    if ((Test-Path -LiteralPath $asy) -and (Test-Is637 $asy)) {
        Say "  Done. The Asylum Igor.exe now reports $(Get-IgorVersion $asy). Original kept as $BackupName." 'Green'
    } else {
        Say '  The Asylum Igor.exe does not report 6.37. Follow INSTRUMENT_SETUP.md to finish by hand.' 'Red'
        exit 1
    }
} else { Say '  Skipped.' }

# --- 3. SOCKIT --------------------------------------------------------------
Step '3/4  SOCKIT XOP (downloads from WaveMetrics, Igor closed)'
if (Confirm-Step "Install SOCKIT into $asyDir ?") {
    Assert-IgorClosed
    if (-not $uv) { $uv = Find-Uv }
    & $uv run --extra hardware scripts\install_sockit.py --download --no-igor --folder $asyDir
    if ($LASTEXITCODE -ne 0) { Say '  SOCKIT install failed (see above).' 'Red'; exit 1 }
    Say '  SOCKIT installed.' 'Green'
} else { Say '  Skipped.' }

# --- 4. Igor procedures -----------------------------------------------------
Step '4/4  The program''s Igor procedures (Igor closed)'
if (Confirm-Step "Install the procedures into $asyDir ?") {
    Assert-IgorClosed
    if (-not $uv) { $uv = Find-Uv }
    & $uv run --extra hardware scripts\install_afm_ipf.py --folder $asyDir
    if ($LASTEXITCODE -ne 0) { Say '  Procedure install failed (see above).' 'Red'; exit 1 }
    Say '  Procedures installed.' 'Green'
} else { Say '  Skipped.' }

Say ''
if ($DryRun) { Say 'Dry run finished. Nothing was changed.' 'Green'; exit 0 }
Say 'Setup finished. Next: double-click "Start Instrument.cmd".' 'Green'
