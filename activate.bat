@echo off
REM ============================================================
REM  Use this instead of .venv\Scripts\activate.bat
REM  It also adds the LAMMPS bin directory to PATH so that
REM  liblammps.dll's dependent DLLs (libgcc, libstdc++, etc.)
REM  are found at runtime.
REM
REM  Usage (in CMD):
REM    activate.bat
REM ============================================================

SET LAMMPS_BIN=C:\Users\kabdelma\AppData\Local\LAMMPS 64-bit 11Feb2026\bin
SET PATH=%LAMMPS_BIN%;%PATH%

call "%~dp0.venv\Scripts\activate.bat"

echo.
echo [AlGe env] Python: & python --version
echo [AlGe env] LAMMPS bin: %LAMMPS_BIN%
echo.
