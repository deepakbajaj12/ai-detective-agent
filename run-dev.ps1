<#
Run both backend (Flask) and frontend (React) for AI Detective Agent.
Usage examples:
  powershell -ExecutionPolicy Bypass -File .\run-dev.ps1
  .\run-dev.ps1 -NoBackend
  .\run-dev.ps1 -NoFrontend
Parameters:
  -NoBackend      Skip starting the Flask backend
  -NoFrontend     Skip starting the React frontend
  -Alpha <float>  Override composite alpha (future use: exposed via env)
#>
param(
  [switch]$NoBackend,
  [switch]$NoFrontend,
  [double]$Alpha = 0.7
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
Write-Host "[AI Detective] Root: $root" -ForegroundColor Cyan

function Ensure-Venv {
  if (-not (Test-Path "$root/venv")) {
    Write-Host "[AI Detective] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv "$root/venv"
  }
}

function Install-BackendDeps {
  if (Test-Path "$root/requirements.txt") {
    Write-Host "[AI Detective] Installing backend dependencies (if needed)..." -ForegroundColor Yellow
    & "$root/venv/Scripts/Activate.ps1"; pip install -q -r "$root/requirements.txt"
  }
}

function Start-Backend {
  Write-Host "[AI Detective] Starting backend (Flask) on http://localhost:5000" -ForegroundColor Green
  $cmd = "cd `"$root`"; & `"$root/venv/Scripts/Activate.ps1`"; $env:COMPOSITE_ALPHA=$Alpha; python app.py"
  Start-Process -FilePath "pwsh" -ArgumentList "-NoLogo","-NoExit","-Command", $cmd | Out-Null
}

function Start-Frontend {
  $fe = Join-Path $root 'detective-frontend'
  $pkg = Join-Path $fe 'package.json'
  if (-not (Test-Path $pkg)) {
    Write-Warning "Frontend package.json not found at $pkg -- skipping frontend startup."
    return
  }
  Write-Host "[AI Detective] Installing frontend dependencies (if needed)..." -ForegroundColor Yellow
  Push-Location $fe
  if (-not (Test-Path 'node_modules')) { npm install --no-audit --no-fund } else { Write-Host "[AI Detective] node_modules present (skipping install)" -ForegroundColor DarkGray }
  Write-Host "[AI Detective] Starting frontend on http://localhost:3000" -ForegroundColor Green
  $cmd = "cd `"$fe`"; npm start"
  Start-Process -FilePath "pwsh" -ArgumentList "-NoLogo","-NoExit","-Command", $cmd | Out-Null
  Pop-Location
}

if (-not $NoBackend) {
  Ensure-Venv
  Install-BackendDeps
  Start-Backend
} else {
  Write-Host "[AI Detective] Skipping backend (NoBackend specified)." -ForegroundColor DarkYellow
}

if (-not $NoFrontend) {
  Start-Frontend
} else {
  Write-Host "[AI Detective] Skipping frontend (NoFrontend specified)." -ForegroundColor DarkYellow
}

Write-Host "[AI Detective] Dev environment launch complete." -ForegroundColor Cyan
