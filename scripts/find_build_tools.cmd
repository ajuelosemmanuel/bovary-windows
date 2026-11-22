@echo off
rem Called by build.cmd after vcvars64. Prefer tools already on PATH.
set "CMAKE_EXE="
set "NINJA_EXE="
for %%i in (cmake.exe) do set "CMAKE_EXE=%%~$PATH:i"
for %%i in (ninja.exe) do set "NINJA_EXE=%%~$PATH:i"
if not defined CMAKE_EXE if exist "%~1\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" set "CMAKE_EXE=%~1\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if not defined NINJA_EXE if exist "%~1\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" set "NINJA_EXE=%~1\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
if not defined CMAKE_EXE (
    echo CMake not found. Add cmake.exe to PATH or install the Visual Studio C++ CMake tools component. 1>&2
    exit /b 1
)
if not defined NINJA_EXE (
    echo Ninja not found. Add ninja.exe to PATH or install the Visual Studio C++ CMake tools component. 1>&2
    exit /b 1
)
exit /b 0
