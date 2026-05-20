@echo off
REM ============================================================
REM  AlGe Phase Diagram — Environment Setup (Windows, uses uv)
REM
REM  Requires: uv  (already installed)
REM  Python 3.11 is downloaded automatically by uv.
REM
REM  Run once from the project directory:
REM    setup_env.bat
REM
REM  Then activate with:
REM    activate.bat
REM ============================================================
setlocal

SET PROJECT_DIR=%~dp0
SET VENV_DIR=%PROJECT_DIR%.venv
SET LAMMPS_PYTHON=C:\Users\kabdelma\AppData\Local\LAMMPS 64-bit 11Feb2026\Python

echo.
echo === Creating Python 3.11 virtual environment with uv ===
uv venv --python 3.11 "%VENV_DIR%"
if errorlevel 1 (
    echo ERROR: uv venv failed. Make sure uv is in PATH.
    exit /b 1
)

SET UPY=%VENV_DIR%\Scripts\python.exe

echo.
echo === Installing fairchem-core and fairchem-lammps (torch pulled automatically) ===
uv pip install fairchem-core fairchem-lammps
if errorlevel 1 (
    echo ERROR: fairchem install failed.
    exit /b 1
)

echo.
echo === Installing remaining requirements and ray[serve] ===
uv pip install -r "%PROJECT_DIR%requirements.txt"
uv pip install "ray[serve]>=2.53.0" torchtnt numba "monty>=2026.2.18" "clusterscope==0.0.18" orjson submitit wandb websockets

echo.
echo === Registering installed LAMMPS Python module via .pth file ===
"%UPY%" -c "import site, os; sp=[p for p in site.getsitepackages() if 'site-packages' in p][0]; open(os.path.join(sp,'lammps_local.pth'),'w').write(r'%LAMMPS_PYTHON%'); print('pth written to', sp)"

echo.
echo ============================================================
echo  Setup complete!
echo.
echo  Activate the environment:
echo    activate.bat
echo.
echo  Validate the setup:
echo    python 00_test_uma_lammps.py
echo ============================================================
endlocal
