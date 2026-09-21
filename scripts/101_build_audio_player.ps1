param(
    [ValidateSet("Tone", "Wav", "Mp3Bypass", "FullStem")]
    [string]$Mode = "Wav",
    [string]$XsaPath = "",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$vitis = "C:\AMDDesignTools\2026.1\Vitis\bin\vitis.bat"
if (-not (Test-Path -LiteralPath $vitis)) {
    throw "Vitis 2026.1 was not found at $vitis"
}

$env:AUDIO_PLAYER_MODE = $Mode
if ($XsaPath) {
    $env:AUDIO_PLAYER_XSA = (Resolve-Path -LiteralPath $XsaPath).Path
} else {
    Remove-Item Env:AUDIO_PLAYER_XSA -ErrorAction SilentlyContinue
}
$env:AUDIO_PLAYER_CLEAN = if ($Clean) { "1" } else { "0" }

$application = switch ($Mode) {
    "Tone" { "tone_player" }
    "Wav" { "wav_player" }
    "Mp3Bypass" { "mp3_bypass_player" }
    "FullStem" { "stem_player" }
}
$buildRoot = Join-Path $repoRoot "hardware\build\navigator_audio_player"
$elf = Join-Path $buildRoot "$application.elf"
$manifest = Join-Path $buildRoot "${application}_build.json"
# Vitis 2026.1's batch launcher can return zero even when its Python script
# raises.  Remove only the two exact generated receipts before invocation and
# require a per-run token in the newly written manifest; stale ELFs can then
# never be mistaken for a successful build.
if (Test-Path -LiteralPath $elf) { Remove-Item -LiteralPath $elf -Force }
if (Test-Path -LiteralPath $manifest) { Remove-Item -LiteralPath $manifest -Force }
$buildToken = [Guid]::NewGuid().ToString("N")
$env:AUDIO_PLAYER_BUILD_TOKEN = $buildToken

& $vitis -s (Join-Path $repoRoot "scripts\100_create_audio_vitis_workspace.py")
if ($LASTEXITCODE -ne 0) {
    throw "Navigator audio player Vitis build failed for mode $Mode"
}

if (-not (Test-Path -LiteralPath $elf -PathType Leaf)) {
    throw "Vitis returned success without producing $elf"
}
if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
    throw "Vitis returned success without producing $manifest"
}
$receipt = Get-Content -Raw -LiteralPath $manifest | ConvertFrom-Json
if ($receipt.build_token -ne $buildToken) {
    throw "Vitis build receipt is stale or belongs to another invocation"
}
Write-Host "AUDIO_PLAYER_BUILD PASS mode=$Mode elf=$elf"
