param(
    [string]$Out = '',
    [string]$Python = 'C:\Users\30519\.workbuddy\binaries\python\envs\stem-npu\Scripts\python.exe',
    [switch]$ProbeOnly,
    [switch]$LaunchWorker,
    [string]$Stamp = ''
)
$ErrorActionPreference = 'Stop'
# The validated WMI host is Windows PowerShell 5.1. A Core caller exports a
# different module/runtime environment: its WMI child failed before receipts
# even in the CPU-only probe. Fail before creation instead of silent fallback.
if ($PSVersionTable.PSEdition -ne 'Desktop' -or $PSVersionTable.PSVersion.Major -ne 5) {
    throw 'Run this launcher with Windows PowerShell 5.1: powershell -NoProfile -File scripts/140_start_paired_teacher_detached.ps1'
}
$pairRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pairResults = Join-Path $pairRoot 'results'
if (-not $Out) { $Out = Join-Path $pairResults 'teacher_pairs_htdemucs_20261002' }
$pairOut = (Resolve-Path -LiteralPath $Out).Path
if (-not $pairOut.StartsWith($pairResults + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Output must be below this workspace results directory.'
}
if (-not (Test-Path -LiteralPath (Join-Path $pairOut 'plan.json') -PathType Leaf)) { throw 'Seal matched plan first.' }
$pairPython = (Resolve-Path -LiteralPath $Python).Path
. (Join-Path $PSScriptRoot '141_process_exit_capture.ps1')

function Write-NewPairReceipt([string]$Path, $Document) {
    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
    $writer = New-Object System.IO.StreamWriter($stream, (New-Object System.Text.UTF8Encoding($false)))
    try { $writer.Write(($Document | ConvertTo-Json -Depth 8)) }
    finally { $writer.Dispose() }
}

function Get-PairJobMembership {
    if (-not ('PairedTeacherDetachNative' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class PairedTeacherDetachNative {
    [DllImport("kernel32.dll")]
    public static extern IntPtr GetCurrentProcess();
    [DllImport("kernel32.dll", SetLastError=true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool IsProcessInJob(IntPtr process, IntPtr job,
        [MarshalAs(UnmanagedType.Bool)] out bool result);
}
'@
    }
    $inJob = $false
    if (-not [PairedTeacherDetachNative]::IsProcessInJob(
        [PairedTeacherDetachNative]::GetCurrentProcess(), [IntPtr]::Zero, [ref]$inJob)) {
        throw 'Cannot inspect launcher job membership.'
    }
    return $inJob
}

function Assert-NoOtherPairWorker {
    $active = @(Get-CimInstance Win32_Process | Where-Object {
        ($_.Name -like 'python*' -and $_.CommandLine -match '(134_generate_teacher_library|139_generate_paired_htdemucs)\.py"?\s+run') -or
        ($_.ProcessId -ne $PID -and $_.Name -like '*powershell*' -and
            $_.CommandLine -like '*140_start_paired_teacher_detached.ps1*' -and
            $_.CommandLine -like '*-LaunchWorker*' -and $_.CommandLine -like ('*' + $pairOut + '*'))
    })
    if ($active.Count) { throw 'Another teacher worker/launcher is active; no duplicate launch.' }
}

if ($LaunchWorker) {
    if ($Stamp -notmatch '^\d{8}_\d{6}_\d{7}$') { throw 'Invalid receipt stamp.' }
    $receipt = Join-Path $pairOut "detached_launch_${Stamp}.json"
    $exitReceipt = Join-Path $pairOut "detached_exit_${Stamp}.json"
    $stdout = Join-Path $pairOut "worker_${Stamp}.stdout.log"
    $stderr = Join-Path $pairOut "worker_${Stamp}.stderr.log"
    foreach ($target in @($receipt, $exitReceipt, $stdout, $stderr)) {
        if (Test-Path -LiteralPath $target) { throw 'Existing launch evidence preserved.' }
    }
    $self = Get-CimInstance Win32_Process -Filter "ProcessId = $PID"
    $parent = Get-CimInstance Win32_Process -Filter "ProcessId = $($self.ParentProcessId)"
    $doc = [ordered]@{
        schema = 1; utc = [DateTime]::UtcNow.ToString('o'); launch_pid = $PID
        parent_pid = $self.ParentProcessId; parent_name = $parent.Name; in_job_object = Get-PairJobMembership
        probe_only = [bool]$ProbeOnly; status = 'checking_detachment'; worker = $null; error = $null
        launcher_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash
    }
    try {
        if ($doc.in_job_object -or $doc.parent_name -ine 'WmiPrvSE.exe') {
            throw 'Not detached under WMI service; refusing CUDA launch.'
        }
        $doc.ffmpeg = (Get-Command ffmpeg -ErrorAction Stop).Source
        $doc.ffprobe = (Get-Command ffprobe -ErrorAction Stop).Source
        if ($ProbeOnly) {
            $doc.status = 'probe_passed'
            Write-NewPairReceipt $receipt $doc
            exit 0
        }
        Assert-NoOtherPairWorker
        $script = Join-Path $PSScriptRoot '139_generate_paired_htdemucs.py'
        $arguments = @('-u', ('"' + $script + '"'), 'run', '--out', ('"' + $pairOut + '"'))
        $worker = Start-Process -FilePath $pairPython -ArgumentList $arguments -WorkingDirectory $pairRoot -WindowStyle Hidden `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
        $retainedWorkerHandle = Hold-PairedProcessHandle $worker
        $doc.worker = [ordered]@{ LauncherProcessId = $worker.Id; Stdout = $stdout; Stderr = $stderr
            Status = (Join-Path $pairOut 'run_status.json') }
        $doc.status = 'worker_started_not_completed'
        Write-NewPairReceipt $receipt $doc
        # Independent helper retains venv parent and records the real launcher's exit.
        $workerExitCode = Wait-PairedProcessExit $worker $retainedWorkerHandle
        Write-NewPairReceipt $exitReceipt ([ordered]@{ schema = 1; utc = [DateTime]::UtcNow.ToString('o')
            launch_receipt = $receipt; launcher_process_id = $worker.Id; exit_code = $workerExitCode
            exit_capture_sha256 = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot '141_process_exit_capture.ps1') -Algorithm SHA256).Hash })
        exit $workerExitCode
    } catch {
        if (-not (Test-Path -LiteralPath $receipt)) {
            $doc.status = 'failed'; $doc.error = $_.Exception.Message
            Write-NewPairReceipt $receipt $doc
        } else {
            Write-NewPairReceipt (Join-Path $pairOut "detached_helper_error_${Stamp}.json") `
                ([ordered]@{ utc = [DateTime]::UtcNow.ToString('o'); error = $_.Exception.Message })
        }
        exit 1
    }
}

if (-not $ProbeOnly) { Assert-NoOtherPairWorker }
$pairStamp = [DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss_fffffff')
$pairShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$command = '"' + $pairShell + '" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' +
    $PSCommandPath + '" -LaunchWorker -Out "' + $pairOut + '" -Python "' + $pairPython + '" -Stamp "' + $pairStamp + '"'
if ($ProbeOnly) { $command += ' -ProbeOnly' }
$environment = [string[]]@(Get-ChildItem Env: | ForEach-Object { $_.Name + '=' + $_.Value })
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{
    ShowWindow = [uint16]0
    CreateFlags = [uint32](16777216 -bor 1024 -bor 16)
    EnvironmentVariables = $environment
}
$created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $command; CurrentDirectory = $pairRoot; ProcessStartupInformation = $startup
}
if ($created.ReturnValue -ne 0) { throw ('WMI launch failed code=' + $created.ReturnValue + '; no fallback/retry.') }
[pscustomobject]@{
    DetachedLauncherPid = $created.ProcessId
    Receipt = (Join-Path $pairOut "detached_launch_${pairStamp}.json")
    ExitReceipt = (Join-Path $pairOut "detached_exit_${pairStamp}.json")
    ProbeOnly = [bool]$ProbeOnly
}
