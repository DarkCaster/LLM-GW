#Requires -Version 5

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$BuildId,
    [Parameter(Position = 1)]
    [string]$BuildVariant,
    [Parameter(Position = 2)]
    [string]$BaseDir = $null,
    [Parameter(Position = 3)]
    [bool]$DlCudaLibs = $false
)

function Show-Usage {
    Write-Host "Simple script for downloading llama.cpp builds for windows from github"
    Write-Host "USAGE:"
    Write-Host "  .\download-llama-cpp.ps1 <build_id> <build_variant> [base_dir] [download_cuda_libs: 0|1]"
    Write-Host ""
    Write-Host "PARAMETERS:"
    Write-Host "  build_id           - Build number (e.g., 7833)"
    Write-Host "  build_variant      - Build variant (cuda-13.1-x64, cuda-12.4-x64, cpu-arm64, cpu-x64, hip-radeon-x64)"
    Write-Host "  base_dir           - Optional base directory for extraction (defaults to script directory)"
    Write-Host "  download_cuda_libs - Optional - include cuda libs download if present for selected build variant, 1 or 0, default: 0 (do not download)"
    Write-Host ""
    Write-Host "EXAMPLES:"
    Write-Host "  .\download-llama-cpp.ps1 7833 cuda-12.4-x64"
    Write-Host "  .\download-llama-cpp.ps1 7833 cpu-x64 C:\Tools"
    Write-Host ""
}

if ([string]::IsNullOrEmpty($BuildId) -or [string]::IsNullOrEmpty($BuildVariant)) {
    Show-Usage
    exit 1
}

if ([string]::IsNullOrEmpty($BaseDir)) {
    $BaseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}

$downloadUrl = "https://github.com/ggml-org/llama.cpp/releases/download/b${BuildId}/llama-b${BuildId}-bin-win-${BuildVariant}.zip"
$zipFile = Join-Path $BaseDir "llama-b${BuildId}-bin-win-${BuildVariant}.zip"
$extractPath = Join-Path $BaseDir "llama-b${BuildId}-${BuildVariant}"

$cudaLibs = ""
if ($BuildVariant -eq "cuda-13.1-x64" -or $BuildVariant -eq "cuda-12.4-x64") {
    $cudaLibs=$BuildVariant
}
$cudaLibsDownloadUrl = "https://github.com/ggml-org/llama.cpp/releases/download/b${BuildId}/cudart-llama-bin-win-${BuildVariant}.zip"
$cudaLibsZipFile = Join-Path $BaseDir "cudart-${BuildVariant}.zip"

Write-Host "Build ID: $BuildId"
Write-Host "Build Variant: $BuildVariant"
Write-Host "Download URL: $downloadUrl"
Write-Host "Base Directory: $BaseDir"
if ($DlCudaLibs) {
    Write-Host "CudaLibs: $cudaLibs"
}
Write-Host ""

Write-Host "Downloading llama.cpp build..."
$ProgressPreference = 'SilentlyContinue'
try {
    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipFile -UseBasicParsing
    Write-Host "Download completed"
} catch {
    Write-Host "Error downloading file: $_"
    exit 1
}

Write-Host "Extracting archive..."
if (Test-Path $extractPath) {
    Write-Host "Removing existing extraction directory..."
    Remove-Item -Path $extractPath -Recurse -Force
}

try {
    Expand-Archive -Path $zipFile -DestinationPath $extractPath -Force
    Write-Host "Extraction completed to: $extractPath"
} catch {
    Write-Host "Error extracting archive: $_"
    Remove-Item -Path $zipFile -Force -ErrorAction SilentlyContinue
    exit 1
}

if ($DlCudaLibs) {
    Write-Host "Downloading cuda libs"
   $ProgressPreference = 'SilentlyContinue'
    try {
        Invoke-WebRequest -Uri $cudaLibsDownloadUrl -OutFile $cudaLibsZipFile -UseBasicParsing
        Write-Host "Download completed"
    } catch {
        Write-Host "Error downloading file: $_"
        exit 1
    }
    try {
        Expand-Archive -Path $cudaLibsZipFile -DestinationPath $extractPath -Force
        Write-Host "Extraction completed to: $extractPath"
    } catch {
        Write-Host "Error extracting archive: $_"
        Remove-Item -Path $cudaLibsZipFile -Force -ErrorAction SilentlyContinue
        exit 1
    }
}

Write-Host "Cleaning up..."
Remove-Item -Path $zipFile -Force
if ($DlCudaLibs) {
    Remove-Item -Path $cudaLibsZipFile -Force
}

Write-Host "Done"
Write-Host "Files extracted to: $extractPath"
