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

& $vitis -s (Join-Path $repoRoot "scripts\100_create_audio_vitis_workspace.py")
if ($LASTEXITCODE -ne 0) {
    throw "Navigator audio player Vitis build failed for mode $Mode"
}

$application = switch ($Mode) {
    "Tone" { "tone_player" }
    "Wav" { "wav_player" }
    "Mp3Bypass" { "mp3_bypass_player" }
    "FullStem" { "stem_player" }
}
$buildRoot = Join-Path $repoRoot "hardware\build\navigator_audio_player"
$elf = Join-Path $buildRoot "$application.elf"
$manifest = Join-Path $buildRoot "${application}_build.json"
if (-not (Test-Path -LiteralPath $elf -PathType Leaf)) {
    throw "Vitis returned success without producing $elf"
}
if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
    throw "Vitis returned success without producing $manifest"
}
Write-Host "AUDIO_PLAYER_BUILD PASS mode=$Mode elf=$elf"
