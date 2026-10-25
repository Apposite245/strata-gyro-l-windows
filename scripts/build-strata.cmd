@echo off
rem Builds Strata's engine (AgentionAI's Gyro-L branch) against AgentionAI's llama.cpp fork, CUDA, plus the parity tools.
rem Run scripts\get-sources.ps1 first. Edit the three paths below for your machine.
rem
rem CUDA 13.0-13.2 accept the Visual Studio 2019/2022 compilers only. With VS 2026 Build Tools installed, the v143
rem (VS 2022) toolset inside it works: vcvars64.bat -vcvars_ver=14.44. With VS 2022, drop -vcvars_ver.

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "VCVER=-vcvars_ver=14.44"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
rem 120 = RTX 50 series. 89 = RTX 40, 86 = RTX 30 (untested here). RTX 50 needs CUDA 13: 12.8 crashes the prompt path.
set "CUDA_ARCH=120"

set "ROOT=%~dp0.."
call "%VCVARS%" %VCVER% >nul || exit /b 1
set "PATH=%CUDA_PATH%\bin;%PATH%"
cd /d "%ROOT%\strata" || exit /b 1
if not exist build\build.ninja (
  rem the options of Strata's own setup.py (build_engine), plus the tests for the parity tools
  cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON ^
    -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=%CUDA_ARCH% ^
    -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" -DCUDAToolkit_ROOT="%CUDA_PATH%" ^
    -DSTRATA_GGML_DIR="%ROOT%\agention-llama.cpp" || exit /b 1
)
cmake --build build --target strata native_expert_parity tq_cpu_parity && goto ok
rem Strata's setup retries once: CUDA 13.0's ptxas now and then fails on a file it just wrote
echo (the build stopped - trying it once more)
cmake --build build --target strata native_expert_parity tq_cpu_parity || exit /b 1
:ok
echo Built: %ROOT%\strata\build\strata.exe
