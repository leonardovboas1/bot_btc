param([string]$Script, [string]$ScriptArgs = "", [string]$Log)
$root = "C:\Users\leona\Desktop\bot_trade"
$py = Join-Path $root ".venv\Scripts\python.exe"
foreach ($v in "OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS") { Set-Item "Env:$v" "1" }
$argl = "$Script $ScriptArgs"
$p = Start-Process -FilePath $py -ArgumentList $argl -WorkingDirectory $root -PassThru -NoNewWindow `
     -RedirectStandardOutput $Log -RedirectStandardError "$Log.err"
$mine = @($p.Id)
while (-not $p.HasExited) {
    # my processes: the launched one + its descendants (loky workers) + anything whose command line mentions research\a8
    $all = Get-CimInstance Win32_Process -Filter "Name='python.exe'"
    foreach ($q in $all) {
        if ($mine -contains $q.ParentProcessId -or $q.CommandLine -match "research.a8") { if ($mine -notcontains $q.ProcessId) { $mine += $q.ProcessId } }
    }
    foreach ($id in $mine) {
        $pr = Get-Process -Id $id -ErrorAction SilentlyContinue
        if ($pr) { try { $pr.ProcessorAffinity = 15; $pr.PriorityClass = 'Idle' } catch {} }
    }
    $free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    if ($free -lt 6) {
        "RAM watchdog: free $free GB < 6, stopping" | Out-File -Append "$Log.watchdog"
        foreach ($id in $mine) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }
        break
    }
    Start-Sleep -Seconds 10
}
"exit $($p.ExitCode)" | Out-File -Append "$Log.watchdog"
