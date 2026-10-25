# Clones the two source trees this build uses, pinned to the commits it was tested with.
# Usage:  .\scripts\get-sources.ps1 [-Root <folder>]      (default: the repository folder)
param([string]$Root = (Split-Path $PSScriptRoot -Parent))

$ErrorActionPreference = "Stop"
$sources = @(
    # AgentionAI's Strata branch: Strata 0.2.4 + the Gyro-L (pair trellis) kernels
    @{ Dir = "strata"; Url = "https://gitlab.com/agentionai-group/strata.git"; Commit = "22dfa634ef3bbac5ad2b1a555f50e9857938f19b" },
    # AgentionAI's llama.cpp fork from GITLAB: its ggml has TQP5/6/7. The GitHub copy does not (see the README).
    @{ Dir = "agention-llama.cpp"; Url = "https://gitlab.com/agentionai-group/llama.cpp.git"; Commit = "9aa51ce4fdd3435a91531ffe16a17414ccc6dc6f" }
)

foreach ($s in $sources) {
    $dir = Join-Path $Root $s.Dir
    if (-not (Test-Path $dir)) {
        # core.autocrlf=false: a Windows checkout with autocrlf=true changes text files the build hashes
        git -c core.autocrlf=false init -q $dir
        git -C $dir config core.autocrlf false
        git -C $dir remote add origin $s.Url
    }
    git -C $dir fetch -q --depth 1 origin $s.Commit
    git -C $dir checkout -q --detach FETCH_HEAD
    Write-Host ("{0}: {1}" -f $s.Dir, (git -C $dir log --oneline -1))
}

$count = Select-String (Join-Path $Root "agention-llama.cpp\ggml\include\ggml.h") -Pattern 'GGML_TYPE_COUNT\s*=\s*(\d+)'
if ($count.Matches[0].Groups[1].Value -ne "150") { throw "the llama.cpp fork has no TQP types (GGML_TYPE_COUNT != 150)" }
Write-Host "ggml has the TQP types (GGML_TYPE_COUNT = 150)"
