# End-to-end demo of the live data plane against mock backends (simulated
# timing, placeholder tokens). Run from the repository root:
#
#   powershell -ExecutionPolicy Bypass -File scripts\demo.ps1
#
# Steps: build; start two mock backends and the control plane (127.0.0.1:18080,
# 18100, 18101); stream one request; kill a backend and watch ejection and
# retries; send a burst that makes the scaler start a third backend through the
# process executor; shut everything down. Logs go to outputs\demo\.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$out = 'outputs\demo'
$cp = 'http://127.0.0.1:18080'
New-Item -ItemType Directory -Force -Path 'outputs\bin', $out | Out-Null
Remove-Item "$out\*.log", "$out\*.out", "$out\*.json" -ErrorAction SilentlyContinue

function Wait-Http([string]$url, [int]$attempts = 100) {
    for ($i = 0; $i -lt $attempts; $i++) {
        try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 $url | Out-Null; return } catch { Start-Sleep -Milliseconds 100 }
    }
    throw "timed out waiting for $url"
}

# Request bodies go through files: Windows PowerShell 5.1 strips quotes from
# arguments passed to native programs.
function Write-Body([string]$name, [string]$json) {
    $path = Join-Path $out $name
    [IO.File]::WriteAllText((Join-Path (Get-Location) $path), $json)
    return $path
}

Write-Output '== build'
go build -o outputs\bin\mock-backend.exe .\cmd\mock-backend
if ($LASTEXITCODE -ne 0) { throw 'build failed' }
go build -o outputs\bin\control-plane.exe .\cmd\control-plane
if ($LASTEXITCODE -ne 0) { throw 'build failed' }

$procs = @()
$scaled = $false
$complete = 0
try {
    Write-Output '== start mock backends and the control plane'
    $m0 = Start-Process -FilePath outputs\bin\mock-backend.exe -ArgumentList '-listen', '127.0.0.1:18100', '-model', 'chat-8b', '-class', 'h100-8b' -RedirectStandardError "$out\mock-0.log" -NoNewWindow -PassThru
    $m1 = Start-Process -FilePath outputs\bin\mock-backend.exe -ArgumentList '-listen', '127.0.0.1:18101', '-model', 'chat-8b', '-class', 'h100-8b' -RedirectStandardError "$out\mock-1.log" -NoNewWindow -PassThru
    $procs += $m0, $m1
    Wait-Http 'http://127.0.0.1:18100/health'
    Wait-Http 'http://127.0.0.1:18101/health'
    $cpProc = Start-Process -FilePath outputs\bin\control-plane.exe -ArgumentList '-config', 'configs\live.json' -RedirectStandardError "$out\control-plane.log" -NoNewWindow -PassThru
    $procs += $cpProc
    Wait-Http "$cp/health"

    Write-Output '== 1. one streamed request (server-sent events through the proxy)'
    $b = Write-Body 'stream.json' '{"model":"chat-8b","stream":true,"messages":[{"role":"user","content":"Hello"}],"mock_output_tokens":8}'
    curl.exe -s -m 30 -N -X POST "$cp/v1/chat/completions" -H 'Content-Type: application/json' --data-binary "@$b" -o "$out\stream.out"
    $lines = Get-Content "$out\stream.out" | Where-Object { $_ -like 'data: *' }
    Write-Output "data events received: $($lines.Count)"
    $lines | Select-Object -First 1
    $lines | Select-Object -Last 1

    Write-Output '== 2. kill mock-0; requests keep succeeding (passive ejection, retries before the first byte)'
    Stop-Process -Id $m0.Id -Force
    $b = Write-Body 'small.json' '{"model":"chat-8b","prompt":"Hello there","max_tokens":4}'
    for ($i = 1; $i -le 6; $i++) {
        $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 30 -Method Post -ContentType 'application/json' -InFile $b "$cp/v1/completions"
        Write-Output "request ${i}: HTTP $($r.StatusCode) via $($r.Headers['X-Replica'])"
    }
    (Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 "$cp/admin/replicas").Content
    (Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 "$cp/metrics").Content -split "`n" | Where-Object { $_ -match '^lsc_(retries_total|upstream_errors_total|replica_healthy)' }

    Write-Output '== 3. a burst of 30 long streams; the threshold scaler adds a replica through the process executor'
    $b = Write-Body 'burst.json' '{"model":"chat-8b","stream":true,"mock_prompt_tokens":200,"mock_output_tokens":600}'
    $burst = @()
    for ($i = 1; $i -le 30; $i++) {
        $burst += Start-Process -FilePath curl.exe -ArgumentList '-s', '-m', '120', '-N', '-X', 'POST', "$cp/v1/chat/completions", '-H', 'Content-Type: application/json', '--data-binary', "@$b", '-o', "$out\burst-$i.out" -NoNewWindow -PassThru
    }
    for ($i = 0; $i -lt 120 -and -not $scaled; $i++) {
        $reps = (Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 "$cp/admin/replicas").Content | ConvertFrom-Json
        if ($reps | Where-Object { $_.id -eq 'h100-8b-p01' -and $_.state -eq 'ready' }) { $scaled = $true } else { Start-Sleep -Milliseconds 500 }
    }
    Write-Output "scale-out observed: $scaled"
    (Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 "$cp/admin/replicas").Content
    $burst | Wait-Process -Timeout 150 -ErrorAction SilentlyContinue
    $complete = @(Get-ChildItem "$out\burst-*.out" | Where-Object { Select-String -Path $_.FullName -SimpleMatch 'data: [DONE]' -Quiet }).Count
    Write-Output "burst streams completed: $complete/30"

    Write-Output '== 4. control-plane metrics'
    (Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 "$cp/metrics").Content -split "`n" | Where-Object { $_ -match '^lsc_requests_total' }
}
finally {
    Write-Output '== shut down'
    try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 -Method Post "$cp/admin/shutdown" | Out-Null } catch { }
    Start-Sleep -Seconds 1
    foreach ($p in $procs) { if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } }
}
if ($scaled -and $complete -eq 30) { Write-Output 'demo OK' } else { Write-Output 'demo FAILED'; exit 1 }
