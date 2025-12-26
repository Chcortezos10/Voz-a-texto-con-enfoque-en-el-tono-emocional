@echo off
echo ===================================================
echo   INICIANDO SISTEMA VOZ-A-TEXTO UNIFICADO (V3)
echo ===================================================
echo.
echo Iniciando API Unificada (Puerto 8000)...
start /min cmd /k "uvicorn app_fastapi:app --host 127.0.0.1 --port 8000"

echo.
echo [OK] Backend cargado.
echo      - API: http://127.0.0.1:8000
echo.
echo Abriendo Dashboard...
start dashboard.html
pause
