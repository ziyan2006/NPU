param(
    [Parameter(Mandatory=$true)][string]$Activation,
    [Parameter(Mandatory=$true)][string]$ExpectedActivationSha,
    [string]$Python = 'C:\Users\30519\.workbuddy\binaries\python\envs\stem-npu\Scripts\python.exe',
    [switch]$LaunchWorker,
    [string]$Stamp = ''
)
$ErrorActionPreference = 'Stop'
if ($PSVersionTable.PSEdition -ne 'Desktop' -or $PSVersionTable.PSVersion.Major -ne 5) { throw 'WindowsPowerShell5.1 required; no fallback.' }
$emaLaunchRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$emaLaunchActivation = (Resolve-Path -LiteralPath $Activation).Path
$emaLaunchPython = (Resolve-Path -LiteralPath $Python).Path
$emaLaunchOut = Join-Path $emaLaunchRoot 'results\mel_ema_single_trajectory_20261005'
$emaLaunchReceipts = Join-Path $emaLaunchRoot 'results\mel_ema_single_trajectory_launch_20261005'
if ($ExpectedActivationSha -notmatch '^[0-9a-f]{64}$' -or (Get-FileHash -LiteralPath $emaLaunchActivation -Algorithm SHA256).Hash -ine $ExpectedActivationSha) { throw 'Activation SHA changed.' }
if (-not $emaLaunchActivation.StartsWith((Join-Path $emaLaunchRoot 'results')+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Activation outside scoped results.' }
foreach ($emaLaunchArgument in @($emaLaunchRoot,$emaLaunchActivation,$emaLaunchPython,$PSCommandPath)) {
    if ($emaLaunchArgument.Contains('"') -or $emaLaunchArgument.Contains("`r") -or $emaLaunchArgument.Contains("`n")) { throw 'Unrepresentable fixed launch token.' }
}
if ((Get-FileHash -LiteralPath (Join-Path $PSScriptRoot '141_process_exit_capture.ps1') -Algorithm SHA256).Hash -ine 'fdd6a7bf690623472f0d270dc6762807a3ca52f08c0a99bb7e08b151f409fde3') { throw 'Original native exit primitive changed.' }
. (Join-Path $PSScriptRoot '141_process_exit_capture.ps1')
function Write-NewEmaReceipt([string]$Path, $Document) {
    $stream = [IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    $writer = New-Object IO.StreamWriter($stream,(New-Object Text.UTF8Encoding($false)))
    try { $writer.Write(($Document | ConvertTo-Json -Depth 12)) } finally { $writer.Dispose() }
}
function Assert-NoEmaDuplicate {
    $tasks = @(Get-CimInstance Win32_Process | Where-Object {
        ($_.Name -like 'python*' -and ($_.CommandLine -like ('*'+$emaLaunchRoot+'*') -or $_.CommandLine -match 'scripts[\\/](?:_test_ema|_review_ema|(?:19[3-9]|20[0-9]|21[0-9]|22[0-9])_)')) -or
        ($_.ProcessId -ne $PID -and $_.Name -like '*powershell*' -and $_.CommandLine -match '224_start_mel_ema_single_trajectory\.ps1' -and $_.CommandLine -like '*-LaunchWorker*')
    })
    if ($tasks.Count) { throw 'Another scoped Python/helper task exists; no duplicate.' }
    if ((Get-PSDrive -Name D).Free -lt 12GB) { throw 'Disk reserve below12GiB.' }
}
function Ema-InJob {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class EmaLaunchNative {
    [DllImport("kernel32.dll")] public static extern IntPtr GetCurrentProcess();
    [DllImport("kernel32.dll", SetLastError=true)]
    [return: MarshalAs(UnmanagedType.Bool)] public static extern bool IsProcessInJob(IntPtr p, IntPtr j, [MarshalAs(UnmanagedType.Bool)] out bool result);
}
'@
    $inJob = $false
    if (-not [EmaLaunchNative]::IsProcessInJob([EmaLaunchNative]::GetCurrentProcess(),[IntPtr]::Zero,[ref]$inJob)) { throw 'Native job membership unavailable.' }
    return $inJob
}
Assert-NoEmaDuplicate
if (Test-Path -LiteralPath $emaLaunchOut) { throw 'Formal run already exists; preserve, never restart.' }
if ($LaunchWorker) {
    if ($Stamp -notmatch '^\d{8}_\d{6}_\d{7}$') { throw 'Receipt stamp required.' }
    $receipt = Join-Path $emaLaunchReceipts "detached_launch_${Stamp}.json"
    $exitReceipt = Join-Path $emaLaunchReceipts "detached_exit_${Stamp}.json"
    $stdout = Join-Path $emaLaunchReceipts "supervisor_${Stamp}.stdout.log"
    $stderr = Join-Path $emaLaunchReceipts "supervisor_${Stamp}.stderr.log"
    foreach ($target in @($receipt,$exitReceipt,$stdout,$stderr)) { if (Test-Path -LiteralPath $target) { throw 'Existing launch evidence.' } }
    $self = Get-CimInstance Win32_Process -Filter "ProcessId = $PID"
    $parent = Get-CimInstance Win32_Process -Filter "ProcessId = $($self.ParentProcessId)"
    $doc = [ordered]@{schema=1;utc=[DateTime]::UtcNow.ToString('o');purpose='NONRELEASE_EMA221_SINGLE_TRAJECTORY_FIXED500';launch_pid=$PID;parent_pid=$self.ParentProcessId;parent_name=$parent.Name;in_job_object=Ema-InJob;activation=$emaLaunchActivation;activation_sha256=$ExpectedActivationSha;out=$emaLaunchOut;launcher_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash;status='checking_actual_detachment';worker=$null;error=$null}
    try {
        if ($doc.in_job_object -or $doc.parent_name -ine 'WmiPrvSE.exe') { throw 'WMI service detached helper required.' }
        $arguments = @('-B','-u',('"'+(Join-Path $PSScriptRoot '221_ema_supervised_training.py')+'"'),'train','--activation',('"'+$emaLaunchActivation+'"'),'--out',('"'+$emaLaunchOut+'"'),'--device','cuda')
        $worker = Start-Process -FilePath $emaLaunchPython -ArgumentList $arguments -WorkingDirectory $emaLaunchRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
        $retainedHandle = Hold-PairedProcessHandle $worker
        $doc.worker = [ordered]@{LauncherProcessId=$worker.Id;Stdout=$stdout;Stderr=$stderr;Status=(Join-Path $emaLaunchOut 'run_status.json');NativeJournal=(Join-Path $emaLaunchOut 'native_event_journal.jsonl')}
        $doc.status='native_supervisor_started_not_completed'
        Write-NewEmaReceipt $receipt $doc
        $exitCode = Wait-PairedProcessExit $worker $retainedHandle
        Write-NewEmaReceipt $exitReceipt ([ordered]@{schema=1;utc=[DateTime]::UtcNow.ToString('o');launch_receipt=$receipt;activation_sha256=$ExpectedActivationSha;launcher_process_id=$worker.Id;exit_code=$exitCode;exit_capture_sha256='fdd6a7bf690623472f0d270dc6762807a3ca52f08c0a99bb7e08b151f409fde3'})
        exit $exitCode
    } catch {
        if (-not(Test-Path -LiteralPath $receipt)) { $doc.status='failed';$doc.error=$_.Exception.Message;Write-NewEmaReceipt $receipt $doc }
        else { Write-NewEmaReceipt (Join-Path $emaLaunchReceipts "detached_helper_error_${Stamp}.json") ([ordered]@{utc=[DateTime]::UtcNow.ToString('o');error=$_.Exception.Message}) }
        exit 1
    }
}
if (Test-Path -LiteralPath $emaLaunchReceipts) { throw 'A launch directory exists; preserve and inspect, no retry.' }
# New metadata/resource-only preflight; it never invokes a worker or old CLI.
& $emaLaunchPython -B (Join-Path $PSScriptRoot '225_prepare_mel_ema_activation.py') check --activation $emaLaunchActivation
if ($LASTEXITCODE -ne 0) { throw 'Complete activation/resources/duplicates rejected; not launching.' }
New-Item -ItemType Directory -Path $emaLaunchReceipts | Out-Null
$emaLaunchStamp = [DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss_fffffff')
$emaLaunchShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$command = '"'+$emaLaunchShell+'" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "'+$PSCommandPath+'" -LaunchWorker -Activation "'+$emaLaunchActivation+'" -ExpectedActivationSha "'+$ExpectedActivationSha+'" -Python "'+$emaLaunchPython+'" -Stamp "'+$emaLaunchStamp+'"'
$environment = [string[]]@(Get-ChildItem Env: | ForEach-Object {$_.Name+'='+$_.Value})
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0;CreateFlags=[uint32](16777216 -bor 1024 -bor 16);EnvironmentVariables=$environment}
$created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command;CurrentDirectory=$emaLaunchRoot;ProcessStartupInformation=$startup}
Write-NewEmaReceipt (Join-Path $emaLaunchReceipts "wmi_request_${emaLaunchStamp}.json") ([ordered]@{schema=1;utc=[DateTime]::UtcNow.ToString('o');return_value=$created.ReturnValue;detached_launcher_pid=$created.ProcessId;activation=$emaLaunchActivation;activation_sha256=$ExpectedActivationSha;command=$command;launch_receipt=(Join-Path $emaLaunchReceipts "detached_launch_${emaLaunchStamp}.json");exit_receipt=(Join-Path $emaLaunchReceipts "detached_exit_${emaLaunchStamp}.json")})
if ($created.ReturnValue -ne 0) { throw ('Actual WMI failure '+$created.ReturnValue+'; no fallback/retry.') }
[pscustomobject]@{DetachedLauncherPid=$created.ProcessId;Receipt=(Join-Path $emaLaunchReceipts "detached_launch_${emaLaunchStamp}.json");ExitReceipt=(Join-Path $emaLaunchReceipts "detached_exit_${emaLaunchStamp}.json")}
