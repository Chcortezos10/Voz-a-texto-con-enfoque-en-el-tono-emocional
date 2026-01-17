@echo off
chcp 65001 >nul
echo ===================================================
echo   INICIANDO SISTEMA VOZ-A-TEXTO EMOCIONAL (V5)
echo   Dashboard Monolitico + API FastAPI
echo ===================================================
echo.

REM Verificar que estamos en el directorio correcto
cd /d "%~dp0"

echo [*] Verificando dependencias...
python -c "import fastapi, uvicorn, whisper, librosa" 2>nul
if errorlevel 1 (
    echo [!] Faltan dependencias. Ejecutando pip install...
    pip install -r requirements.txt
)

echo.
echo [*] Iniciando API Unificada (Puerto 8000)...
echo     Modelo Whisper: medium (GPU RTX 4050)
echo.
echo.

REM Iniciar servidor con optimizaciones de memoria
start /min cmd /k "python -m uvicorn app_fastapi:app --host 127.0.0.1 --port 8000 --workers 1"

REM Esperar a que el servidor inicie
echo [*] Esperando a que el servidor inicie...
timeout /t 5 /nobreak >nul

REM Verificar que el servidor está corriendo
curl -s http://127.0.0.1:8000/health >nul 2>&1
if errorlevel 1 (
    echo [!] El servidor puede tardar mas en cargar modelos...
    timeout /t 10 /nobreak >nul
)

echo.
echo ===================================================
echo [OK] Sistema iniciado correctamente!
echo ===================================================
echo.
echo      API:       http://127.0.0.1:8000
echo      Docs:      http://127.0.0.1:8000/docs
echo      Dashboard: dashboard.html
echo.

REM Abrir dashboard monolitico directamente en el navegador
echo [*] Abriendo Dashboard en el navegador...
if exist "dashboard.html" (
    start "" "%~dp0dashboard.html"
) else (
    echo [!] No se encontro dashboard.html
    echo     Por favor verifica que el archivo existe
)

echo.
echo Presiona cualquier tecla para cerrar este mensaje...
echo (El servidor seguira corriendo en segundo plano)
pause >nul
