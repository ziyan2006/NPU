param(
    [string]$Out = '',
    [string]$Python = 'C:\Users\30519\.workbuddy\binaries\python\envs\stem-npu\Scripts\python.exe',
    [switch]$ProbeOnly,
    [switch]$LaunchWorker,
    [string]$LaunchReceipt = ''
)
$ErrorActionPreference = 'Stop'
$detachRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$detachResults = Join-Path $detachRoot 'results'
if (-not $Out) { $Out = Join-Path $detachResults 'teacher_library_melband_20261002' }
$detachOut = (Resolve-Path -LiteralPath $Out).Path
if (-not $detachOut.StartsWith($detachResults + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Output must be below the workspace results directory.'
}
if (-not (Test-Path -LiteralPath (Join-Path $detachOut 'plan.json') -PathType Leaf)) {
    throw 'A sealed teacher plan is required.'
}
$detachPython = (Resolve-Path -LiteralPath $Python).Path

function Write-NewDetachReceipt([string]$Path, $Document) {
    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
    $writer = New-Object System.IO.StreamWriter($stream, (New-Object System.Text.UTF8Encoding($false)))
    try { $writer.Write(($Document | ConvertTo-Json -Depth 8)) }
    finally { $writer.Dispose() }
}

function Get-DetachJobMembership {
    if (-not ('TeacherDetachNative' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class TeacherDetachNative {
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
    if (-not [TeacherDetachNative]::IsProcessInJob(
        [TeacherDetachNative]::GetCurrentProcess(), [IntPtr]::Zero, [ref]$inJob)) {
        throw ('Cannot inspect process job membership: ' + [Runtime.InteropServices.Marshal]::GetLastWin32Error())
    }
    return $inJob
}

if ($LaunchWorker) {
    $receiptTarget = [System.IO.Path]::GetFullPath($LaunchReceipt)
    if ((Split-Path -Parent $receiptTarget) -ne $detachOut -or
        (Split-Path -Leaf $receiptTarget) -notmatch '^detached_launch_\d{8}_\d{6}_\d{7}\.json$') {
        throw 'Invalid launch receipt target.'
    }
    if (Test-Path -LiteralPath $receiptTarget) { throw 'Refusing to overwrite a launch receipt.' }
    $self = Get-CimInstance Win32_Process -Filter "ProcessId = $PID"
    $parent = Get-CimInstance Win32_Process -Filter "ProcessId = $($self.ParentProcessId)"
    $doc = [ordered]@{
        schema = 1
        utc = [DateTime]::UtcNow.ToString('o')
        launch_pid = $PID
        parent_pid = $self.ParentProcessId
        parent_name = $parent.Name
        in_job_object = Get-DetachJobMembership
        probe_only = [bool]$ProbeOnly
        status = 'checking_detachment'
        worker = $null
        error = $null
    }
    try {
        if ($doc.in_job_object -or $doc.parent_name -ine 'WmiPrvSE.exe') {
            throw 'Launcher is not detached from process jobs under the WMI service; refusing to start CUDA.'
        }
        # Prove decoder discovery without decoding, loading models, or querying CUDA.
        $doc.ffmpeg = (Get-Command ffmpeg -ErrorAction Stop).Source
        $doc.ffprobe = (Get-Command ffprobe -ErrorAction Stop).Source
        if ($ProbeOnly) {
            $doc.status = 'probe_passed'
        } else {
            # Keep the existing inference/resume recipe unchanged. It rechecks every saved song.
            $doc.worker = & (Join-Path $PSScriptRoot '135_start_teacher_library.ps1') -Out $detachOut -Python $detachPython
            $doc.status = 'worker_started_not_completed'
        }
    } catch {
        $doc.status = 'failed'
        $doc.error = $_.Exception.Message
        Write-NewDetachReceipt $receiptTarget $doc
        exit 1
    }
    Write-NewDetachReceipt $receiptTarget $doc
    exit 0
}

# A new WMI host process is the parent, not this shell or the Codex runtime.
# No OS scheduled task, startup hook, registry change, or automatic restart loop is installed.
if (-not $ProbeOnly) {
    $active = @(Get-CimInstance Win32_Process | Where-Object {
        ($_.Name -like 'python*' -and $_.CommandLine -like '*134_generate_teacher_library.py*' -and
            $_.CommandLine -like ('*' + $detachOut + '*')) -or
        ($_.ProcessId -ne $PID -and $_.Name -like '*powershell*' -and
            $_.CommandLine -like '*137_start_teacher_library_detached.ps1*' -and
            $_.CommandLine -like '*-LaunchWorker*' -and $_.CommandLine -like ('*' + $detachOut + '*'))
    })
    if ($active.Count) { throw 'A matching worker/launcher is already active; refusing duplicate launch.' }
}
$detachStamp = [DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss_fffffff')
$detachReceipt = Join-Path $detachOut "detached_launch_${detachStamp}.json"
$detachShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$command = '"' + $detachShell + '" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' +
    $PSCommandPath + '" -LaunchWorker -Out "' + $detachOut + '" -Python "' +
    $detachPython + '" -LaunchReceipt "' + $detachReceipt + '"'
if ($ProbeOnly) { $command += ' -ProbeOnly' }
# Preserve the caller's decoding/runtime environment, but never log its values.
$environment = [string[]]@(Get-ChildItem Env: | ForEach-Object { $_.Name + '=' + $_.Value })
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{
    ShowWindow = [uint16]0
    # Windows PowerShell 5.1 needs a console; ShowWindow=0 keeps the new console hidden.
    # CREATE_BREAKAWAY_FROM_JOB | CREATE_UNICODE_ENVIRONMENT | CREATE_NEW_CONSOLE
    CreateFlags = [uint32](16777216 -bor 1024 -bor 16)
    EnvironmentVariables = $environment
}
$created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $command
    CurrentDirectory = $detachRoot
    ProcessStartupInformation = $startup
}
if ($created.ReturnValue -ne 0) {
    throw ('Windows WMI refused independent launch, code=' + $created.ReturnValue + '; no fallback launch attempted.')
}
[pscustomobject]@{
    DetachedLauncherPid = $created.ProcessId
    Receipt = $detachReceipt
    ProbeOnly = [bool]$ProbeOnly
    Note = 'Check receipt plus run_status/logs; a created process is not proof of completed inference.'
}
