#!/usr/bin/env python3
"""
🚀 EJECUTOR OPTIMIZADO FLAN - EXPERIMENTO COMPLETO
Versión optimizada con DescentEnv REAL + mejoras automáticas
"""

import os
import time
import sys
import json
import psutil
from datetime import datetime, timedelta
from contextlib import redirect_stdout, redirect_stderr

def print_system_info():
    """Imprime información del sistema para optimización"""
    print("🖥️  INFORMACIÓN DEL SISTEMA")
    print("=" * 50)
    print(f"CPU cores: {psutil.cpu_count(logical=False)} físicos, {psutil.cpu_count(logical=True)} lógicos")
    print(f"RAM disponible: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print(f"Uso actual CPU: {psutil.cpu_percent(interval=1)}%")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def estimate_completion_time():
    """Estima tiempo de finalización"""
    print("⏱️  ESTIMACIÓN DE TIEMPO")
    print("=" * 50)
    print("📊 Episodios totales: 256,000")
    print("🔄 Factor BlueSky: 30x más lento que Mock")
    print("⏰ Tiempo estimado: 30-35 horas")
    
    start_time = datetime.now()
    estimated_end = start_time + timedelta(hours=32.5)
    
    print(f"🕐 Inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🏁 Finalización estimada: {estimated_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 Duración: {estimated_end - start_time}")
    print()

def create_monitoring_script():
    """Crea script de monitoreo en tiempo real"""
    monitoring_script = '''#!/bin/bash
# Script de monitoreo del experimento
echo "🔍 MONITOREO EXPERIMENTO FLAN"
echo "=========================="

while true; do
    echo "$(date): Revisando progreso..."
    
    # Verificar archivos de salida
    if [ -f "flan_results_256k_ultra.json" ]; then
        echo "✅ Archivo de resultados encontrado"
        SIZE=$(ls -lh flan_results_256k_ultra.json | awk '{print $5}')
        echo "📁 Tamaño: $SIZE"
    fi
    
    # Verificar CPU y memoria
    echo "💻 CPU: $(top -l 1 -n 0 | grep "CPU usage" | awk '{print $3}' | sed 's/%//')"
    echo "💾 RAM: $(top -l 1 -n 0 | grep "PhysMem" | awk '{print $2}')"
    
    # Verificar modelos guardados
    if [ -d "models_ultramega_256k_ultra" ]; then
        echo "🎯 Modelos encontrados: $(ls models_ultramega_256k_ultra/ | wc -l)"
    fi
    
    echo "------------------------"
    sleep 300  # Revisar cada 5 minutos
done
'''
    
    with open('monitor_experimento.sh', 'w') as f:
        f.write(monitoring_script)
    
    os.chmod('monitor_experimento.sh', 0o755)
    print("📊 Script de monitoreo creado: monitor_experimento.sh")

def run_experiment():
    """Ejecuta el experimento completo"""
    print("🚀 INICIANDO EXPERIMENTO COMPLETO")
    print("=" * 80)
    
    # Verificar que estamos usando DescentEnv REAL
    print("🔍 Verificando configuración...")
    
    try:
        # Importar para verificar
        from descent_env import DescentEnv
        print("✅ DescentEnv REAL confirmado")
    except ImportError:
        print("❌ ERROR: DescentEnv no disponible")
        return False
    
    # Ejecutar experimento principal
    print("\n🏃 Ejecutando flan_qlearning_solution.py...")
    
    start_time = time.time()
    
    try:
        # Importar y ejecutar el experimento
        from flan_qlearning_solution import main
        
        print("🎯 Iniciando experimento con mejoras automáticas...")
        results = main()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n✅ EXPERIMENTO COMPLETADO")
        print(f"⏱️ Tiempo total: {total_time / 3600:.1f} horas")
        print(f"📊 Resultados guardados en: flan_results_256k_ultra.json")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR en experimento: {e}")
        return False

def main():
    """Función principal"""
    print("🎯 FLAN - EJECUTOR OPTIMIZADO")
    print("=" * 80)
    print("🔥 CONFIGURACIÓN: DescentEnv REAL + Mejoras Automáticas")
    print("📈 OBJETIVO: Salto de -81.6 → -25 (mejora de +56.6 puntos)")
    print("⏱️ DURACIÓN ESTIMADA: 30-35 horas")
    print()
    
    # Información del sistema
    print_system_info()
    
    # Estimación de tiempo
    estimate_completion_time()
    
    # Crear script de monitoreo
    create_monitoring_script()
    
    # Confirmación final
    print("🚨 CONFIRMACIÓN FINAL")
    print("=" * 50)
    print("• Entorno: DescentEnv REAL (BlueSky)")
    print("• Mejoras: Automáticas integradas")
    print("• Episodios: 256,000 totales")
    print("• Tiempo: ~30-35 horas")
    print("• Políticas: Científicamente válidas")
    print()
    
    confirm = input("¿Proceder con el experimento? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ Experimento cancelado")
        return
    
    # Ejecutar experimento
    success = run_experiment()
    
    if success:
        print("\n🎉 EXPERIMENTO COMPLETADO EXITOSAMENTE")
        print("📊 Revisa los resultados en flan_results_256k_ultra.json")
        print("🎯 Modelos guardados en models_ultramega_256k_ultra/")
    else:
        print("\n❌ El experimento falló")

if __name__ == "__main__":
    main() 