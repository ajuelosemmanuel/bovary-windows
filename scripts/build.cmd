@echo off
setlocal
rem Usage: scripts\build.cmd [cuda^|cpu] [parallel-jobs] [test-paths]
cd /d "%~dp0.."
rem nvcc's generated #include paths cannot safely use the sandbox TEMP path.
if not exist work\build\tmp mkdir work\build\tmp
set "TEMP=%CD%\work\build\tmp"
set "TMP=%TEMP%"
set "BACKEND=%~1"
if not defined BACKEND set "BACKEND=cpu"
set "JOBS=%~2"
if not defined JOBS set "JOBS=4"
set "PATH_TESTS=%~3"
if defined PATH_TESTS if /i not "%PATH_TESTS%"=="test-paths" (echo Third argument must be test-paths or omitted. & exit /b 1)
if defined PATH_TESTS if /i not "%BACKEND%"=="cpu" (echo test-paths requires the cpu backend. & exit /b 1)
set "ENGINE_DIR=%CD%\work\engine\bovary-llama.cpp"
set "BUILD_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "BUILD_PYTHON=%CD%\.venv\Scripts\python.exe"
"%BUILD_PYTHON%" scripts\bootstrap_engine.py --directory "%ENGINE_DIR%" --check
if errorlevel 1 exit /b 1
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (echo Visual Studio Installer vswhere.exe not found. & exit /b 1)
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%i"
if not defined VSROOT (echo Visual Studio x64 C++ tools not found. & exit /b 1)
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
call scripts\find_build_tools.cmd "%VSROOT%"
if errorlevel 1 exit /b 1
rem Native MSVC SIMD probes can optimize away unused AVX512 operations.
set "CPU_FLAGS=-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF"
if /i "%BACKEND%"=="cpu" goto cpu
if /i not "%BACKEND%"=="cuda" (echo Backend must be cuda or cpu. & exit /b 1)
set "CUDA_PATH=%CD%\work\deps\cuda-13.3"
if not exist "%CUDA_PATH%\bin\nvcc.exe" (echo Run python tools\inventory_cuda_setup.py first. & exit /b 1)
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;%PATH%"
"%CMAKE_EXE%" -S . -B work/build/bovary-cuda -G Ninja "-DBOVARY_ENGINE_DIR:PATH=%ENGINE_DIR%" "-DCMAKE_MAKE_PROGRAM=%NINJA_EXE%" -DCMAKE_BUILD_TYPE=Release -DCMAKE_TRY_COMPILE_CONFIGURATION=Release %CPU_FLAGS% -DBOVARY_CPU_ONLY=OFF -DGGML_CUDA=ON -DGGML_STATIC=ON -DCMAKE_CUDA_ARCHITECTURES=86 "-DCUDAToolkit_ROOT=%CUDA_PATH%" "-DCMAKE_CUDA_COMPILER=%CUDA_PATH%\bin\nvcc.exe"
if errorlevel 1 exit /b 1
"%CMAKE_EXE%" --build work/build/bovary-cuda --config Release --parallel %JOBS%
exit /b %errorlevel%
:cpu
"%CMAKE_EXE%" -S . -B work/build/bovary-cpu -G Ninja "-DBOVARY_ENGINE_DIR:PATH=%ENGINE_DIR%" "-DCMAKE_MAKE_PROGRAM=%NINJA_EXE%" -DCMAKE_BUILD_TYPE=Release -DCMAKE_TRY_COMPILE_CONFIGURATION=Release %CPU_FLAGS% -DBOVARY_CPU_ONLY=ON -DGGML_CUDA=OFF
if errorlevel 1 exit /b 1
"%CMAKE_EXE%" --build work/build/bovary-cpu --config Release --parallel %JOBS%
if errorlevel 1 exit /b 1
if defined PATH_TESTS "%CMAKE_EXE%" --build work/build/bovary-cpu --config Release --target bovary-paths-test --parallel %JOBS%
exit /b %errorlevel%
