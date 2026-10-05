param([string]$Out = '')
$ErrorActionPreference = 'Stop'
if ($PSVersionTable.PSEdition -ne 'Desktop') { throw 'Use Windows PowerShell 5.1 for this reproduction.' }
. (Join-Path $PSScriptRoot '141_process_exit_capture.ps1')
$exitTestRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$exitTestResults = Join-Path $exitTestRoot 'results'
if (-not $Out) { $Out = Join-Path $exitTestResults 'process_exit_capture_probe_20261002' }
$exitTestOut = [System.IO.Path]::GetFullPath($Out)
if (-not $exitTestOut.StartsWith($exitTestResults + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Probe output must be below workspace results.'
}
if (Test-Path -LiteralPath $exitTestOut) { throw 'Existing CPU probe evidence preserved.' }
New-Item -ItemType Directory -Path $exitTestOut | Out-Null
$exitTestShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$exitTestRows = @()
foreach ($expected in @(0, 7)) {
    foreach ($capture in @($false, $true)) {
        $ident = 'code_' + $expected + '_capture_' + $capture
        $p = Start-Process -FilePath $exitTestShell -ArgumentList @('-NoProfile', '-NonInteractive', '-Command', ('"exit ' + $expected + '"')) `
            -WindowStyle Hidden -RedirectStandardOutput (Join-Path $exitTestOut ($ident + '.stdout.log')) `
            -RedirectStandardError (Join-Path $exitTestOut ($ident + '.stderr.log')) -PassThru
        if ($capture) {
            $retained = Hold-PairedProcessHandle $p
            $actual = Wait-PairedProcessExit $p $retained
            if ($actual -ne $expected) { throw ('Captured wrong exit code: ' + $actual) }
        } else {
            $p.WaitForExit()
            $p.Refresh()
            $actual = $p.ExitCode
        }
        $exitTestRows += [pscustomobject]@{ expected = $expected; handle_retained = $capture; actual = $actual }
        $p.Dispose()
    }
}
try { Wait-PairedProcessExit ([System.Diagnostics.Process]::GetCurrentProcess()) ([IntPtr]::Zero) | Out-Null; throw 'Missing handle was accepted.' }
catch { if ($_.Exception.Message -ne 'A retained process handle is required.') { throw } }
$doc = [ordered]@{ schema = 1; powershell_version = $PSVersionTable.PSVersion.ToString(); rows = $exitTestRows
    zero_and_nonzero_captured = $true; missing_handle_rejected = $true; teacher_inference = $false
    module_sha256 = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot '141_process_exit_capture.ps1') -Algorithm SHA256).Hash }
$json = $doc | ConvertTo-Json -Depth 6
$stream = [System.IO.File]::Open((Join-Path $exitTestOut 'probe.json'), [System.IO.FileMode]::CreateNew)
$writer = New-Object System.IO.StreamWriter($stream, (New-Object System.Text.UTF8Encoding($false)))
try { $writer.Write($json) } finally { $writer.Dispose() }
Write-Output $json
Write-Output 'PROCESS_EXIT_CAPTURE_CPU PASS'
