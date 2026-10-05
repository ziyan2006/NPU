param(
    [string]$Out = '',
    [string]$Python = 'C:\Users\30519\.workbuddy\binaries\python\envs\stem-npu\Scripts\python.exe',
    [switch]$ProbeOnly,
    [switch]$LaunchWorker,
    [switch]$Resume,
    [string]$Stamp = ''
)
$ErrorActionPreference = 'Stop'
if ($PSVersionTable.PSEdition -ne 'Desktop' -or $PSVersionTable.PSVersion.Major -ne 5) {
    throw 'Use Windows PowerShell 5.1; no Core launch fallback.'
}
$melLRScaleRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$melLRScaleResults = Join-Path $melLRScaleRoot 'results'
if (-not $Out) { $Out = Join-Path $melLRScaleResults 'mel_lr_scale_20261004' }
$melLRScaleOut = [System.IO.Path]::GetFullPath($Out)
if (-not $melLRScaleOut.StartsWith($melLRScaleResults + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Run output must be a child of workspace results.'
}
$melLRScaleReceipts = Join-Path $melLRScaleResults 'mel_lr_scale_launch_20261004'
$melLRScalePython = (Resolve-Path -LiteralPath $Python).Path
. (Join-Path $PSScriptRoot '141_process_exit_capture.ps1')

function Write-NewMelLRScaleReceipt([string]$Path, $Document) {
    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
    $writer = New-Object System.IO.StreamWriter($stream, (New-Object System.Text.UTF8Encoding($false)))
    try { $writer.Write(($Document | ConvertTo-Json -Depth 12)) }
    finally { $writer.Dispose() }
}

function Assert-NoMelLRScaleConflict {
    $active = @(Get-CimInstance Win32_Process | Where-Object {
        ($_.Name -like 'python*' -and $_.CommandLine -match '(186_train_mel_lf_boundary|188_review_mel_lf_boundary|191_diagnose_train_lf_cross_boundary|192_diagnose_train_lf_cross_boundary_recovery|196_review_mel_lr_scale|134_generate_teacher_library|139_generate_paired_htdemucs|150_train_paired_exploration|154_train_mel_weak_weight|161_train_mel_source_aux|166_train_mel_source_strength|172_train_mel_instrumental_protection|169_diagnose_gradient_contributions|175_diagnose_auxiliary_components|178_train_mel_component_ablation|181_diagnose_train_context_reconstruction|182_diagnose_train_context_reconstruction_recovery|184_diagnose_train_low_frequency_reference|194_train_mel_lr_scale)\.py"?\s+(run|train|smoke)') -or
        ($_.ProcessId -ne $PID -and $_.Name -like '*powershell*' -and $_.CommandLine -match '(151_start_paired_exploration|155_start_mel_weak_weight|162_start_mel_source_aux|167_start_mel_source_strength|173_start_mel_instrumental_protection|179_start_mel_component_ablation|195_start_mel_lr_scale|195_start_mel_lr_scale)\.ps1' -and
            $_.CommandLine -like '*-LaunchWorker*' -and $_.CommandLine -notlike '*-ProbeOnly*')
    })
    if ($active.Count) { throw 'Existing teacher/training worker or helper; refusing duplicate launch.' }
}

function Get-MelLRScaleJobMembership {
    if (-not ('MelLRScaleDetachNative' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class MelLRScaleDetachNative {
    [DllImport("kernel32.dll")] public static extern IntPtr GetCurrentProcess();
    [DllImport("kernel32.dll", SetLastError=true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool IsProcessInJob(IntPtr process, IntPtr job,
        [MarshalAs(UnmanagedType.Bool)] out bool result);
}
'@
    }
    $inJob = $false
    if (-not [MelLRScaleDetachNative]::IsProcessInJob([MelLRScaleDetachNative]::GetCurrentProcess(), [IntPtr]::Zero, [ref]$inJob)) {
        throw 'Cannot establish native job membership.'
    }
    return $inJob
}

if (-not $ProbeOnly) {
    Assert-NoMelLRScaleConflict
    foreach ($relative in @('mel_lr_scale_import_20261004\approval.json',
        'mel_lr_scale_cpu_20261004\mechanism.json', 'mel_lr_scale_cuda_20261004\mechanism.json',
        'train_lf_cross_boundary_recovery_20261004\diagnostic.json', 'mel_lr_scale_audit_20261004\audit.json')) {
        if (-not (Test-Path -LiteralPath (Join-Path $melLRScaleResults $relative) -PathType Leaf)) {
            throw ('Missing approval/preflight: ' + $relative)
        }
    }
    if ($Resume) {
        if (-not (Test-Path -LiteralPath $melLRScaleOut -PathType Container)) { throw 'No existing run to resume.' }
    } elseif (Test-Path -LiteralPath $melLRScaleOut) { throw 'Run already exists; preserve it, use explicit resume if unfinished.' }
}

if ($LaunchWorker) {
    if ($Stamp -notmatch '^\d{8}_\d{6}_\d{7}$') { throw 'Invalid receipt stamp.' }
    $receipt = Join-Path $melLRScaleReceipts "detached_launch_${Stamp}.json"
    $exitReceipt = Join-Path $melLRScaleReceipts "detached_exit_${Stamp}.json"
    $stdout = Join-Path $melLRScaleReceipts "worker_${Stamp}.stdout.log"
    $stderr = Join-Path $melLRScaleReceipts "worker_${Stamp}.stderr.log"
    foreach ($target in @($receipt, $exitReceipt, $stdout, $stderr)) {
        if (Test-Path -LiteralPath $target) { throw 'Existing launch evidence preserved.' }
    }
    $self = Get-CimInstance Win32_Process -Filter "ProcessId = $PID"
    $parent = Get-CimInstance Win32_Process -Filter "ProcessId = $($self.ParentProcessId)"
    $doc = [ordered]@{ schema = 1; utc = [DateTime]::UtcNow.ToString('o'); launch_pid = $PID
        parent_pid = $self.ParentProcessId; parent_name = $parent.Name; in_job_object = Get-MelLRScaleJobMembership
        probe_only = [bool]$ProbeOnly; resume = [bool]$Resume; out = $melLRScaleOut
        purpose = 'NONRELEASE_MEL_LR_SCALE_FORK'; teacher = 'kim_melband';
        arm_roles = @{ htdemucs_waveform_control = 'lr1_control'; kim_melband_waveform_candidate = 'lr_half_candidate' }; status = 'checking_detachment'; worker = $null; error = $null
        launcher_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash }
    try {
        if ($doc.in_job_object -or $doc.parent_name -ine 'WmiPrvSE.exe') { throw 'Helper not detached under WMI service.' }
        $doc.ffmpeg = (Get-Command ffmpeg -ErrorAction Stop).Source
        $doc.ffprobe = (Get-Command ffprobe -ErrorAction Stop).Source
        if ($ProbeOnly) {
            $doc.status = 'probe_passed'
            Write-NewMelLRScaleReceipt $receipt $doc
            exit 0
        }
        $script = Join-Path $PSScriptRoot '194_train_mel_lr_scale.py'
        $arguments = @('-u', ('"' + $script + '"'), 'train', '--device', 'cuda', '--approval', ('"' + (Join-Path $melLRScaleResults 'mel_lr_scale_import_20261004\approval.json') + '"'), '--out', ('"' + $melLRScaleOut + '"'))
        if ($Resume) { $arguments += '--resume' }
        $worker = Start-Process -FilePath $melLRScalePython -ArgumentList $arguments -WorkingDirectory $melLRScaleRoot -WindowStyle Hidden `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
        $retainedHandle = Hold-PairedProcessHandle $worker
        $doc.worker = [ordered]@{ LauncherProcessId = $worker.Id; Stdout = $stdout; Stderr = $stderr
            Status = (Join-Path $melLRScaleOut 'run_status.json') }
        $doc.status = 'worker_started_not_completed'
        Write-NewMelLRScaleReceipt $receipt $doc
        $exitCode = Wait-PairedProcessExit $worker $retainedHandle
        Write-NewMelLRScaleReceipt $exitReceipt ([ordered]@{ schema = 1; utc = [DateTime]::UtcNow.ToString('o')
            launch_receipt = $receipt; launcher_process_id = $worker.Id; exit_code = $exitCode
            exit_capture_sha256 = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot '141_process_exit_capture.ps1') -Algorithm SHA256).Hash })
        exit $exitCode
    } catch {
        if (-not (Test-Path -LiteralPath $receipt)) {
            $doc.status = 'failed'; $doc.error = $_.Exception.Message
            Write-NewMelLRScaleReceipt $receipt $doc
        } else {
            Write-NewMelLRScaleReceipt (Join-Path $melLRScaleReceipts "detached_helper_error_${Stamp}.json") `
                ([ordered]@{ utc = [DateTime]::UtcNow.ToString('o'); error = $_.Exception.Message })
        }
        exit 1
    }
}

if (-not (Test-Path -LiteralPath $melLRScaleReceipts)) { New-Item -ItemType Directory -Path $melLRScaleReceipts | Out-Null }
$melLRScaleStamp = [DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss_fffffff')
$melLRScaleShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$command = '"' + $melLRScaleShell + '" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' +
    $PSCommandPath + '" -LaunchWorker -Out "' + $melLRScaleOut + '" -Python "' + $melLRScalePython + '" -Stamp "' + $melLRScaleStamp + '"'
if ($ProbeOnly) { $command += ' -ProbeOnly' }
if ($Resume) { $command += ' -Resume' }
$environment = [string[]]@(Get-ChildItem Env: | ForEach-Object { $_.Name + '=' + $_.Value })
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{
    ShowWindow = [uint16]0; CreateFlags = [uint32](16777216 -bor 1024 -bor 16); EnvironmentVariables = $environment }
$created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $command; CurrentDirectory = $melLRScaleRoot; ProcessStartupInformation = $startup }
if ($created.ReturnValue -ne 0) { throw ('WMI launch failed code=' + $created.ReturnValue + '; no fallback or retry.') }
[pscustomobject]@{ DetachedLauncherPid = $created.ProcessId; Receipt = (Join-Path $melLRScaleReceipts "detached_launch_${melLRScaleStamp}.json")
    ExitReceipt = (Join-Path $melLRScaleReceipts "detached_exit_${melLRScaleStamp}.json"); ProbeOnly = [bool]$ProbeOnly }
