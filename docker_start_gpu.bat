@echo off
chcp 65001 >nul
echo ===================================================
echo   INICIANDO VOZ-A-TEXTO CON DOCKER (GPU ENABLED)
echo ===================================================
echo.

REM Verificar que estamos en el directorio correcto
cd /d "%~dp0"

echo [*] Verificando GPU NVIDIA...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [!] ADVERTENCIA: No se detectó nvidia-smi
    echo     Asegúrate de tener los drivers NVIDIA instalados
    pause
)

echo.
echo [*] Reconstruyendo imagen Docker con GPU...
docker compose build --no-cache

if errorlevel 1 (
    echo [!] Error construyendo la imagen
    pause
    exit /b 1
)

echo.
echo [*] Iniciando contenedor con soporte GPU...
start /b docker compose up

echo.
echo [*] Esperando a que el servidor inicie (puede tardar 30-60 segundos)...
timeout /t 10 /nobreak >nul

:check_server
rem Verificar si el contenedor sigue corriendo
docker ps -q -f name=transcriptor-emocional-v8 | findstr . >nul
if errorlevel 1 (
    echo.
    echo [!] ERROR CRITICO: El contenedor se ha detenido inesperadamente.
    echo     Mostrando logs...
    echo.
    docker compose logs transcriptor-api
    pause
    exit /b 1
)

rem Verificar si el servidor responde
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo     Servidor aún cargando modelos...
    timeout /t 5 /nobreak >nul
    goto check_server
)

echo.
echo ===================================================
echo [OK] Servidor iniciado correctamente!
echo ===================================================
echo.
echo   Dashboard: http://localhost:8000/
echo   API Docs:  http://localhost:8000/docs
echo.

REM Abrir dashboard en el navegador
echo [*] Abriendo Dashboard en el navegador...
start "" "http://localhost:8000/"

echo.
echo Presiona cualquier tecla para ver los logs del contenedor...
echo (Ctrl+C para salir de los logs)
pause >nul
docker compose logs -f
