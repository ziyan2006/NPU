param(
    [string]$Out = '',
    [string]$Python = 'C:\Users\30519\.workbuddy\binaries\python\envs\stem-npu\Scripts\python.exe'
)
$ErrorActionPreference = 'Stop'
$batchRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$resultsRoot = [System.IO.Path]::GetFullPath((Join-Path $batchRoot 'results'))
if (-not $Out) { $Out = Join-Path $resultsRoot 'teacher_library_melband_20261002' }
$batchOut = (Resolve-Path -LiteralPath $Out).Path
if (-not $batchOut.StartsWith($resultsRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Batch output must be below this workspace results directory.'
}
if (-not (Test-Path -LiteralPath (Join-Path $batchOut 'plan.json') -PathType Leaf)) {
    throw 'Seal the plan with script 134 prepare before launching.'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Python executable missing.' }
$statusPath = Join-Path $batchOut 'run_status.json'
if (Test-Path -LiteralPath $statusPath) {
    $batchStatus = Get-Content -Raw -LiteralPath $statusPath | ConvertFrom-Json
    if ($batchStatus.status -in @('running', 'loading_teacher', 'checking_resume')) {
        $owner = Get-Process -Id $batchStatus.pid -ErrorAction SilentlyContinue
        if ($owner -and $owner.ProcessName -like 'python*') {
            throw "Worker already active, PID=$($batchStatus.pid). The Python worker also enforces an exclusive lock."
        }
    }
}
$stamp = [DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss_fffffff')
$stdout = Join-Path $batchOut "worker_${stamp}.stdout.log"
$stderr = Join-Path $batchOut "worker_${stamp}.stderr.log"
if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr)) {
    throw 'New logs required; refusing overwrite.'
}
$script = Join-Path $PSScriptRoot '134_generate_teacher_library.py'
$arguments = @('-u', ('"' + $script + '"'), 'run', '--out', ('"' + $batchOut + '"'))
$worker = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $batchRoot -WindowStyle Hidden `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
[pscustomobject]@{
    LauncherProcessId = $worker.Id
    Stdout = $stdout
    Stderr = $stderr
    Status = $statusPath
    Note = 'Started, not completed; inspect run_status.json. Venv launcher may create a child worker PID.'
}
