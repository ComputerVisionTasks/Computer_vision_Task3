@echo off
title Computer Vision Server (CMake)

echo ========================================
echo   Building with CMake
echo ========================================
echo.


REM Create build directory
if not exist "build" mkdir build
cd build

REM Configure with CMake
echo [1/3] Configuring...
cmake ..

if %errorlevel% neq 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

REM Build
echo [2/3] Building...
cmake --build . --config Release

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

if exist Release\cv_server.exe (
    Release\cv_server.exe
) else if exist cv_server.exe (
    cv_server.exe
) else (
    echo Error: Could not find cv_server.exe
)

pause