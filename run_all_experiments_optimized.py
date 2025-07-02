#!/usr/bin/env python3
"""
Script maestro optimizado para ejecutar ambos proyectos FLAN y BORED
Maximiza el uso de CPU y memoria del MacBook M1 Pro
"""

import os
import sys
import time
import subprocess
import psutil

def setup_environment():
    """Configura el entorno para máximo rendimiento"""
    print("Configurando entorno para máximo rendimiento...")
    
    # Variables de entorno para optimización
    env_vars = {
        'OPENBLAS_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1', 
        'OMP_NUM_THREADS': '1',
        'VECLIB_MAXIMUM_THREADS': '1',
        'NUMEXPR_MAX_THREADS': '1'
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
    
    # Información del sistema
    cpu_count = psutil.cpu_count(logical=True)
    memory_gb = psutil.virtual_memory().total / (1024**3)
    available_gb = psutil.virtual_memory().available / (1024**3)
    
    print(f"Sistema detectado:")
    print(f"  - CPU: {cpu_count} cores (M1 Pro)")
    print(f"  - RAM Total: {memory_gb:.1f}GB")
    print(f"  - RAM Disponible: {available_gb:.1f}GB")
    print(f"  - Paralelización: {cpu_count-1 if cpu_count else 1} procesos simultáneos")
    print()

def run_flan_experiment():
    """Ejecuta el experimento FLAN"""
    print("="*80)
    print("EJECUTANDO PROYECTO FLAN")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Cambiar al directorio de FLAN
        os.chdir('descent-env')
        
        # Ejecutar experimento FLAN
        result = subprocess.run([
            sys.executable, 'run_flan_experiment.py'
        ], capture_output=False, text=True, check=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ FLAN completado en {duration:.1f} segundos ({duration/60:.1f} minutos)")
        
        # Volver al directorio raíz
        os.chdir('..')
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error en FLAN: {e}")
        os.chdir('..')
        return False
    except Exception as e:
        print(f"\n❌ Error inesperado en FLAN: {e}")
        os.chdir('..')
        return False

def run_bored_experiment():
    """Ejecuta el experimento BORED"""
    print("="*80)
    print("EJECUTANDO PROYECTO BORED")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Cambiar al directorio de BORED
        os.chdir('tactix')
        
        # Ejecutar experimento BORED
        result = subprocess.run([
            sys.executable, 'run_bored_experiment.py'
        ], capture_output=False, text=True, check=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ BORED completado en {duration:.1f} segundos ({duration/60:.1f} minutos)")
        
        # Volver al directorio raíz
        os.chdir('..')
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error en BORED: {e}")
        os.chdir('..')
        return False
    except Exception as e:
        print(f"\n❌ Error inesperado en BORED: {e}")
        os.chdir('..')
        return False

def generate_final_report():
    """Genera un reporte final consolidado"""
    print("="*80)
    print("GENERANDO REPORTE FINAL CONSOLIDADO")
    print("="*80)
    
    print("Archivos generados:")
    print()
    
    print("📁 PROYECTO FLAN:")
    flan_files = [
        "descent-env/flan_results.json",
        "descent-env/flan_results.png", 
        "descent-env/models_fina/",
        "descent-env/models_media/",
        "descent-env/models_gruesa/"
    ]
    
    for file_path in flan_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
    
    print()
    print("📁 PROYECTO BORED:")
    bored_files = [
        "tactix/bored_results.json",
        "tactix/bored_results.png",
        "tactix/bored_models.pkl"
    ]
    
    for file_path in bored_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
    
    print()
    print("📊 RESUMEN DE RESULTADOS:")
    print("  - FLAN: Q-Learning y Stochastic Q-Learning con múltiples discretizaciones")
    print("  - BORED: Minimax y Expectimax con alpha-beta pruning y múltiples heurísticas")
    print("  - Ambos proyectos optimizados para máximo rendimiento en M1 Pro")

def main():
    """Función principal"""
    total_start_time = time.time()
    
    print("🚀 EJECUTOR MAESTRO - PROYECTOS FLAN Y BORED")
    print("Optimizado para MacBook M1 Pro")
    print("="*80)
    
    # Configurar entorno
    setup_environment()
    
    # Ejecutar experimentos
    flan_success = run_flan_experiment()
    bored_success = run_bored_experiment()
    
    # Generar reporte final
    generate_final_report()
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print("="*80)
    print("🎯 RESUMEN FINAL")
    print("="*80)
    print(f"⏱️  Tiempo total: {total_duration:.1f} segundos ({total_duration/60:.1f} minutos)")
    print(f"🔬 FLAN: {'✅ Exitoso' if flan_success else '❌ Falló'}")
    print(f"🎮 BORED: {'✅ Exitoso' if bored_success else '❌ Falló'}")
    
    if flan_success and bored_success:
        print("\n🎉 ¡TODOS LOS EXPERIMENTOS COMPLETADOS EXITOSAMENTE!")
        print("📈 Modelos entrenados y resultados listos para análisis")
        return 0
    else:
        print("\n⚠️  Algunos experimentos fallaron. Revisar logs arriba.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 