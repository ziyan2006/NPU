param(
    [string]$BuildRoot = "hardware/build/navigator_audio_boot"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$bootgen = "C:\AMDDesignTools\2026.1\Vivado\bin\bootgen.bat"
$fsbl = Join-Path $repoRoot "hardware/build/navigator_sd_fsbl/zynq_fsbl.elf"
$bitstream = Join-Path $repoRoot "hardware/build/navigator_z7020_audio_export/stem_npu_audio_navigator_z7020.bit"
$expectedFsblSha256 = "B9C732AEF65EFA97AE89B8D5EFBD0A2A3265F2275DF9C91B66ADEAE31A7AB584"
$modes = @(
    @{ Name = "tone"; Bif = "navigator_audio_tone.bif"; Elf = "tone_player.elf"; BuildMode = "Tone" },
    @{ Name = "wav"; Bif = "navigator_audio_wav.bif"; Elf = "wav_player.elf"; BuildMode = "Wav" },
    @{ Name = "mp3_bypass"; Bif = "navigator_audio_mp3_bypass.bif"; Elf = "mp3_bypass_player.elf"; BuildMode = "Mp3Bypass" },
    @{ Name = "stem"; Bif = "navigator_audio_stem.bif"; Elf = "stem_player.elf"; BuildMode = "FullStem" }
)

function Require-File([string]$Path) {
    if (!(Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "missing required audio boot artifact: $Path"
    }
}

function Require-Hash([string]$Path, [string]$Expected, [string]$Label) {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash
    if ($actual -ne $Expected) {
        throw "$Label SHA-256 mismatch: expected $Expected, got $actual"
    }
}

function Read-ImageNames([string]$Text) {
    $matches = [regex]::Matches(
        $Text,
        '(?m)^\s*IMAGE HEADER \(([^\r\n]+)\)\s*$'
    )
    return @($matches | ForEach-Object { $_.Groups[1].Value })
}

Require-File $bootgen
Require-File $fsbl
Require-File $bitstream
Require-Hash $fsbl $expectedFsblSha256 "handoff FSBL"
$gitCommit = (& git -C $repoRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $gitCommit -notmatch '^[0-9a-f]{40}$') {
    throw "could not determine current Git commit"
}
$gitDirty = [bool]((& git -C $repoRoot status --porcelain --untracked-files=no) -join "")

$resolvedBuildRoot = Join-Path $repoRoot $BuildRoot
foreach ($mode in $modes) {
    $bif = Join-Path $repoRoot "software/audio_player/boot/$($mode.Bif)"
    $directory = Join-Path $resolvedBuildRoot $mode.Name
    $boot = Join-Path $directory "BOOT.BIN"
    $readback = Join-Path $directory "BOOT.read.txt"
    $manifestPath = Join-Path $directory "manifest.json"
    $elf = Join-Path $repoRoot "hardware/build/navigator_audio_player/$($mode.Elf)"
    foreach ($path in @($bif, $boot, $readback, $manifestPath, $elf)) {
        Require-File $path
    }

    $report = (& $bootgen -arch zynq -read $boot 2>&1) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        throw "Bootgen could not inspect $($mode.Name) image"
    }
    $names = @(Read-ImageNames $report)
    $expectedNames = @("zynq_fsbl.elf", "stem_npu_audio_navigator_z7020.bit", $mode.Elf)
    if ($names.Count -ne 3 -or (Compare-Object $expectedNames $names -SyncWindow 0)) {
        throw "$($mode.Name) image partitions must be exactly FSBL, audio bitstream, and matching ELF in order; got $($names -join ', ')"
    }
    $partitionNames = @([regex]::Matches(
        $report,
        '(?m)^\s*PARTITION HEADER TABLE \(([^\r\n]+)\)\s*$'
    ) | ForEach-Object { $_.Groups[1].Value })
    $countMatch = [regex]::Match(
        $report,
        'total_images \(0x04\) : 0x([0-9a-fA-F]+)'
    )
    if (!$countMatch.Success -or
        [Convert]::ToUInt32($countMatch.Groups[1].Value, 16) -ne $partitionNames.Count) {
        throw "$($mode.Name) partition count does not match the Bootgen header"
    }
    $unexpectedPartitions = @($partitionNames | Where-Object {
        $_ -ne 'zynq_fsbl.elf.0' -and
        $_ -ne 'stem_npu_audio_navigator_z7020.bit.0' -and
        $_ -notmatch ('^' + [regex]::Escape($mode.Elf) + '\.[0-9]+$')
    })
    if ($unexpectedPartitions.Count -ne 0) {
        throw "$($mode.Name) image contains unexpected partitions: $($unexpectedPartitions -join ', ')"
    }
    if ($report -match '(?i)jtag') {
        throw "$($mode.Name) image unexpectedly depends on JTAG"
    }

    $savedReadback = Get-Content -Raw -LiteralPath $readback
    if ($savedReadback -ne $report) {
        throw "$($mode.Name) saved Bootgen readback is stale"
    }

    $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    if ($manifest.schema -ne "stem-npu-audio-boot-v1" -or
        $manifest.build_mode -ne $mode.BuildMode -or
        $manifest.git_commit -ne $gitCommit -or
        $manifest.git_dirty -ne $gitDirty -or
        $manifest.vivado_version -notmatch '^2026\.1' -or
        $manifest.vitis_version -notmatch '^2026\.1') {
        throw "$($mode.Name) manifest metadata is incomplete"
    }
    $pathChecks = @(
        @{ Actual = $manifest.inputs.fsbl.path; Expected = "hardware/build/navigator_sd_fsbl/zynq_fsbl.elf" },
        @{ Actual = $manifest.inputs.bitstream.path; Expected = "hardware/build/navigator_z7020_audio_export/stem_npu_audio_navigator_z7020.bit" },
        @{ Actual = $manifest.inputs.elf.path; Expected = "hardware/build/navigator_audio_player/$($mode.Elf)" },
        @{ Actual = $manifest.inputs.bif.path; Expected = "software/audio_player/boot/$($mode.Bif)" },
        @{ Actual = $manifest.outputs.boot_bin.path; Expected = "hardware/build/navigator_audio_boot/$($mode.Name)/BOOT.BIN" },
        @{ Actual = $manifest.outputs.readback.path; Expected = "hardware/build/navigator_audio_boot/$($mode.Name)/BOOT.read.txt" }
    )
    foreach ($entry in $pathChecks) {
        if ($entry.Actual -ne $entry.Expected) {
            throw "$($mode.Name) manifest path mismatch: got $($entry.Actual), expected $($entry.Expected)"
        }
    }
    Require-Hash $fsbl $manifest.inputs.fsbl.sha256 "$($mode.Name) FSBL input"
    Require-Hash $bitstream $manifest.inputs.bitstream.sha256 "$($mode.Name) bitstream input"
    Require-Hash $elf $manifest.inputs.elf.sha256 "$($mode.Name) ELF input"
    Require-Hash $bif $manifest.inputs.bif.sha256 "$($mode.Name) BIF input"
    Require-Hash $boot $manifest.outputs.boot_bin.sha256 "$($mode.Name) BOOT.BIN output"
    Require-Hash $readback $manifest.outputs.readback.sha256 "$($mode.Name) readback output"

    $elfText = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($elf))
    $modeMarker = "AUDIO_BUILD_MODE=$($mode.BuildMode)"
    if (!$elfText.Contains($modeMarker)) {
        throw "$($mode.Name) ELF does not contain build mode marker $modeMarker"
    }
}

& python (Join-Path $repoRoot "scripts/96_check_navigator_audio_soc.py")
if ($LASTEXITCODE -ne 0) { throw "audio SoC address/bitstream gate failed" }

foreach ($script in @(
    "79_test_navigator_sd_boot_probe.ps1",
    "82_test_navigator_sd_npu_bit_only.ps1",
    "85_test_navigator_sd_fsbl_probe_widths.ps1",
    "86_test_navigator_sd_coldboot_image.ps1",
    "89_test_navigator_sd_fsbl.ps1",
    "92_test_navigator_sd_boot_probe_id_guard.ps1"
)) {
    & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot $script)
    if ($LASTEXITCODE -ne 0) { throw "legacy cold-boot regression failed: $script" }
}

Write-Output "NAVIGATOR_AUDIO_BOOT_TEST PASS modes=4"
