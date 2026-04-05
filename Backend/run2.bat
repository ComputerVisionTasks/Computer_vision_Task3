@echo off
title Computer Vision Server (CMake + MinGW)

echo ========================================
echo   Building with CMake (MinGW)
echo ========================================
echo.

REM Configure with CMake using a dedicated MinGW build directory
echo [1/3] Configuring...
cmake -S . -B build-mingw -G "MinGW Makefiles"

if %errorlevel% neq 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

REM Build
echo [2/3] Building...
cmake --build build-mingw -j

if %errorlevel% neq 0 (
    echo Build failed!
    pause
    exit /b 1
)

REM Run
echo [3/3] Starting server...
echo.
echo Server running at http://localhost:8080
echo Press Ctrl+C to stop
echo.

if exist build-mingw\cv_server.exe (
    build-mingw\cv_server.exe
) else if exist build-mingw\Release\cv_server.exe (
    build-mingw\Release\cv_server.exe
) else (
    echo Error: Could not find cv_server.exe
)

pause
