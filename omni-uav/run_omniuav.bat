@echo off
setlocal
cd /d "%~dp0"
set "CONDA_FOUND="
where conda.bat >nul 2>nul && set "CONDA_FOUND=1"
if defined CONDA_FOUND (
  call conda activate omniuav
) else (
  for %%D in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\anaconda3") do (
    if exist "%%~D\Scripts\activate.bat" (
      call "%%~D\Scripts\activate.bat" omniuav
      goto :run
    )
  )
  echo [WARN] conda not found, running with current python.
)
:run
python app.py

