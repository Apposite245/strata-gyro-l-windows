# Downloads Gyro-L (pinned revision, SHA-256 checked), builds the pack and runs the checks.
# Usage:  .\scripts\prepare-gyro-l.ps1 -Data D:\strata-data [-SkipDownload] [-NoGpuCheck]
# Needs: the built engine (scripts\build-strata.cmd), `hf` (pip install huggingface_hub), uv or Python 3.10+.
param(
    [Parameter(Mandatory = $true)][string]$Data,
    [string]$Root = (Split-Path $PSScriptRoot -Parent),
    [switch]$SkipDownload,
    [switch]$NoGpuCheck
)

$ErrorActionPreference = "Stop"
$repo = "agentionai/Qwen3.8-Flash-Next-Gyro-GGUF"
$revision = "d149933df39d2337174e393e558131ab3cde3c83"
$file = "Qwen3.8-Flash-Next-Gyro-L-TQ3_0.gguf"
$sha256 = "5428a5646eaeaa56cc00a892f92e7546aeadb255233503969ad25cf48c7f359b"

$strata = Join-Path $Root "strata"
$models = Join-Path $Data "models\gyro-l"
$pack = Join-Path $Data "packs\gyro-l"
$gguf = Join-Path $models $file
New-Item -ItemType Directory -Force $models | Out-Null

# 1. the model: 108 GB, about half of it the n-gram table read from disk for every token - keep it on a fast NVMe
if (-not $SkipDownload) {
    hf download $repo $file mmproj-F16.gguf --revision $revision --local-dir $models
    if ($LASTEXITCODE -ne 0) { throw "download failed" }
}
Write-Host "checking SHA-256 (about 2 minutes) ..."
$h = (Get-FileHash $gguf -Algorithm SHA256).Hash.ToLower()
if ($h -ne $sha256) { throw "SHA-256 mismatch: $h" }

# 2. a Python environment for the pack tools and the server
$py = Join-Path $strata ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    if (Get-Command uv -ErrorAction SilentlyContinue) { uv venv (Join-Path $strata ".venv") --python 3.12 -q }
    else { python -m venv (Join-Path $strata ".venv") }
    if (Get-Command uv -ErrorAction SilentlyContinue) { uv pip install --python $py -r (Join-Path $strata "requirements.txt") -q }
    else { & $py -m pip install -q -r (Join-Path $strata "requirements.txt") }
}

# 3. the pack: dense weights, tokenizer, Hadamard and expert tables (~1.4 GB); the experts stay in the GGUF
$env:STRATA_GGUF_PY = Join-Path $Root "agention-llama.cpp\gguf-py"
& $py (Join-Path $strata "tools\iq_pack.py") --gguf $gguf --out $pack --compat-bf16
if ($LASTEXITCODE -ne 0) { throw "iq_pack failed" }

# 4. checks: CPU kernels always; the GPU ones unless -NoGpuCheck (they load the card briefly)
$bin = Join-Path $strata "build"
& (Join-Path $bin "tq_cpu_parity.exe") | Select-Object -Last 1
if (-not $NoGpuCheck) {
    foreach ($pair in "tqp6/tqp6", "tqp5/tqp7", "tqp7/tqp5") {
        foreach ($mode in "--synthetic", "--synthetic-hadamard") {
            & (Join-Path $bin "native_expert_parity.exe") $mode $pair | Select-Object -Last 1
            if ($LASTEXITCODE -ne 0) { throw "GPU parity failed: $mode $pair" }
        }
    }
    & (Join-Path $bin "native_expert_parity.exe") --load-check $pack $gguf | Select-Object -Last 1
    if ($LASTEXITCODE -ne 0) { throw "load check failed" }
}
Write-Host "Gyro-L is ready: model $gguf, pack $pack"
