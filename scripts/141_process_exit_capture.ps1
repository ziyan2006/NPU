# Windows PowerShell 5.1 Start-Process + stream redirection may return an
# ExitCode of null unless its process handle is retained before waiting.
# Never coerce a missing exit code to success. This module starts no processes.
function Hold-PairedProcessHandle([System.Diagnostics.Process]$Process) {
    $handle = $Process.Handle
    if ($handle -eq [IntPtr]::Zero) { throw 'No child process handle; exit status is not provable.' }
    return $handle
}

function Wait-PairedProcessExit([System.Diagnostics.Process]$Process, [IntPtr]$RetainedHandle) {
    if ($RetainedHandle -eq [IntPtr]::Zero) { throw 'A retained process handle is required.' }
    $Process.WaitForExit()
    $Process.Refresh()
    $code = $Process.ExitCode
    if ($null -eq $code -or -not $Process.HasExited) {
        throw 'Missing child exit code after wait; refusing to report success.'
    }
    return [int]$code
}
