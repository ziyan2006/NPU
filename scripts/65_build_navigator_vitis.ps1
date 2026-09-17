param(
    [string]$XsaPath = "",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$vitis = "C:\AMDDesignTools\2026.1\Vitis\bin\vitis.bat"
if (-not (Test-Path -LiteralPath $vitis)) {
    throw "Vitis 2026.1 was not found at $vitis"
}

if ($XsaPath) {
    $env:NPU_VITIS_XSA = (Resolve-Path -LiteralPath $XsaPath).Path
} else {
    Remove-Item Env:NPU_VITIS_XSA -ErrorAction SilentlyContinue
}
if ($Clean) {
    $env:NPU_VITIS_CLEAN = "1"
} else {
    Remove-Item Env:NPU_VITIS_CLEAN -ErrorAction SilentlyContinue
}

& $vitis -s (Join-Path $repoRoot "scripts\64_create_navigator_vitis_workspace.py")
if ($LASTEXITCODE -ne 0) {
    throw "Navigator Vitis standalone build failed"
}
