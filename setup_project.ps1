# setup_project.ps1 - Sets up project directory tree on Windows PowerShell
Write-Host "Creating project directory structure..." -ForegroundColor Cyan

$dirs = @(
    "data\raw\d1_alibaba2018",
    "data\raw\d2_azure2019",
    "data\raw\d3_alibaba_gpu2020",
    "data\parquet",
    "data\processed",
    "src",
    "results\figures",
    "results\tables"
)

foreach ($d in $dirs) {
    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Force -Path $d | Out-Null
        Write-Host "Created: $d" -ForegroundColor Green
    } else {
        Write-Host "Exists:  $d" -ForegroundColor DarkGray
    }
}

Write-Host "Directory structure verified successfully." -ForegroundColor Cyan
