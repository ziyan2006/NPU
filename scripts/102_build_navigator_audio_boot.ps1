param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$buildRoot = Join-Path $repoRoot "hardware/build/navigator_audio_boot"
$bootgen = "C:\AMDDesignTools\2026.1\Vivado\bin\bootgen.bat"
$fsbl = Join-Path $repoRoot "hardware/build/navigator_sd_fsbl/zynq_fsbl.elf"
$bitstream = Join-Path $repoRoot "hardware/build/navigator_z7020_audio_export/stem_npu_audio_navigator_z7020.bit"
$expectedFsblSha256 = "7B3AD97C0ED47C94533A80B2FB5A3C46F316962C8CB9D9E9AB14483FA56B377E"
$modes = @(
    @{ Name = "tone"; Bif = "navigator_audio_tone.bif"; Elf = "tone_player.elf"; BuildMode = "Tone" },
    @{ Name = "wav"; Bif = "navigator_audio_wav.bif"; Elf = "wav_player.elf"; BuildMode = "Wav" },
    @{ Name = "mp3_bypass"; Bif = "navigator_audio_mp3_bypass.bif"; Elf = "mp3_bypass_player.elf"; BuildMode = "Mp3Bypass" },
    @{ Name = "stem"; Bif = "navigator_audio_stem.bif"; Elf = "stem_player.elf"; BuildMode = "FullStem" }
)

function Require-File([string]$Path) {
    if (!(Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "missing required audio boot input: $Path"
    }
}

function File-Record([string]$Path) {
    $fullPath = [IO.Path]::GetFullPath($Path)
    $repoPrefix = [IO.Path]::GetFullPath($repoRoot).TrimEnd('\') + '\'
    if (!$fullPath.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "manifest input is outside the repository: $fullPath"
    }
    return [ordered]@{
        path = $fullPath.Substring($repoPrefix.Length).Replace('\', '/')
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash
        bytes = (Get-Item -LiteralPath $Path).Length
    }
}

if ($Clean -and (Test-Path -LiteralPath $buildRoot)) {
    $resolvedRepo = [IO.Path]::GetFullPath($repoRoot).TrimEnd('\') + '\'
    $resolvedBuild = [IO.Path]::GetFullPath($buildRoot).TrimEnd('\') + '\'
    $allowedRoot = [IO.Path]::GetFullPath(
        (Join-Path $repoRoot "hardware/build")
    ).TrimEnd('\') + '\'
    if (!$resolvedBuild.StartsWith($allowedRoot, [StringComparison]::OrdinalIgnoreCase) -or
        !$resolvedBuild.StartsWith($resolvedRepo, [StringComparison]::OrdinalIgnoreCase)) {
        throw "refusing to clean audio build path outside the repository build tree: $resolvedBuild"
    }
    Remove-Item -LiteralPath $buildRoot -Recurse -Force
}

Require-File $bootgen
& powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "91_patch_navigator_sd_fsbl_handoff.ps1")
if ($LASTEXITCODE -ne 0) { throw "handoff-capable FSBL generation failed" }
Require-File $fsbl
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $fsbl).Hash -ne $expectedFsblSha256) {
    throw "handoff FSBL SHA-256 does not match the validated reference"
}
Require-File $bitstream

foreach ($mode in $modes) {
    $buildArguments = @(
        "-ExecutionPolicy", "Bypass", "-File",
        (Join-Path $PSScriptRoot "101_build_audio_player.ps1"),
        "-Mode", $mode.BuildMode
    )
    if ($Clean) { $buildArguments += "-Clean" }
    & powershell @buildArguments
    if ($LASTEXITCODE -ne 0) {
        throw "audio application build failed for $($mode.BuildMode)"
    }
}

$gitCommit = (& git -C $repoRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $gitCommit -notmatch '^[0-9a-f]{40}$') {
    throw "could not determine Git commit for audio boot manifest"
}
$gitDirty = [bool]((& git -C $repoRoot status --porcelain --untracked-files=no) -join "")

foreach ($mode in $modes) {
    $directory = Join-Path $buildRoot $mode.Name
    $boot = Join-Path $directory "BOOT.BIN"
    $readbackPath = Join-Path $directory "BOOT.read.txt"
    $manifestPath = Join-Path $directory "manifest.json"
    $bif = Join-Path $repoRoot "software/audio_player/boot/$($mode.Bif)"
    $elf = Join-Path $repoRoot "hardware/build/navigator_audio_player/$($mode.Elf)"
    Require-File $bif
    Require-File $elf
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
    if (Test-Path -LiteralPath $boot) { Remove-Item -LiteralPath $boot -Force }

    Push-Location $repoRoot
    try {
        & $bootgen -arch zynq -image $bif -o $boot -w
        if ($LASTEXITCODE -ne 0 -or !(Test-Path -LiteralPath $boot)) {
            throw "Bootgen failed for $($mode.Name)"
        }
        $readback = (& $bootgen -arch zynq -read $boot 2>&1) -join "`n"
        if ($LASTEXITCODE -ne 0) {
            throw "Bootgen readback failed for $($mode.Name)"
        }
    } finally {
        Pop-Location
    }
    [IO.File]::WriteAllText($readbackPath, $readback, [Text.UTF8Encoding]::new($false))

    $manifest = [ordered]@{
        schema = "stem-npu-audio-boot-v1"
        build_mode = $mode.BuildMode
        git_commit = $gitCommit
        git_dirty = $gitDirty
        vivado_version = "2026.1 SW Build 6511674"
        vitis_version = "2026.1 SW Build 6497934"
        inputs = [ordered]@{
            fsbl = File-Record $fsbl
            bitstream = File-Record $bitstream
            elf = File-Record $elf
            bif = File-Record $bif
        }
        outputs = [ordered]@{
            boot_bin = File-Record $boot
            readback = File-Record $readbackPath
        }
    }
    $json = $manifest | ConvertTo-Json -Depth 6
    [IO.File]::WriteAllText($manifestPath, $json + "`n", [Text.UTF8Encoding]::new($false))
    Write-Output "NAVIGATOR_AUDIO_BOOT_IMAGE mode=$($mode.BuildMode) path=$boot"
}

Write-Output "NAVIGATOR_AUDIO_BOOT_BUILD PASS modes=4 output=$buildRoot"
