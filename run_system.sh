#!/bin/bash
# ===================================================
#   INICIANDO SISTEMA VOZ-A-TEXTO UNIFICADO (V3)
#   Script para Linux/macOS
# ===================================================

echo "==================================================="
echo "  INICIANDO SISTEMA VOZ-A-TEXTO UNIFICADO (V3)"
echo "==================================================="
echo ""

echo "Iniciando API Unificada (Puerto 8000)..."

# Iniciar uvicorn en segundo plano
uvicorn app_fastapi:app --host 127.0.0.1 --port 8000 &
API_PID=$!

echo ""
echo "[OK] Backend cargado (PID: $API_PID)"
echo "     - API: http://127.0.0.1:8000"
echo ""

echo "Abriendo Dashboard..."
# Abrir dashboard en el navegador predeterminado
if command -v xdg-open &> /dev/null; then
    xdg-open dashboard.html 2>/dev/null &  # Linux
elif command -v open &> /dev/null; then
    open dashboard.html &  # macOS
else
    echo "Abre manualmente: dashboard.html"
fi

echo ""
echo "Presiona Ctrl+C para detener el servidor..."
echo ""

# Esperar a que el usuario detenga el script
wait $API_PID
