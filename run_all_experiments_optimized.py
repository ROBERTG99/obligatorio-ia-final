#!/usr/bin/env python3
"""
Script maestro optimizado para ejecutar ambos proyectos FLAN y BORED
Maximiza el uso de CPU y memoria del MacBook M1 Pro
Configurado para usar entorno virtual y DescentEnv real
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
    print(f"  - Paralelización: {cpu_count if cpu_count else 1} procesos simultáneos")
    print()

def check_virtual_environment():
    """Verifica que el entorno virtual esté disponible"""
    venv_path = os.path.join(os.getcwd(), 'proyecto_env')
    venv_python = os.path.join(venv_path, 'bin', 'python3')
    
    if os.path.exists(venv_python):
        print(f"✅ Entorno virtual encontrado: {venv_path}")
        return venv_python
    else:
        print(f"❌ Entorno virtual no encontrado en: {venv_path}")
        print("📝 Ejecute primero: python3 -m venv proyecto_env && source proyecto_env/bin/activate && pip install psutil seaborn pygame bluesky-gym gymnasium")
        return None

def run_flan_experiment(python_exe):
    """Ejecuta el experimento FLAN con DescentEnv real"""
    print("="*80)
    print("EJECUTANDO PROYECTO FLAN con DescentEnv Real")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Cambiar al directorio de FLAN
        original_dir = os.getcwd()
        os.chdir('descent-env')
        
        print("🚀 Iniciando experimento FLAN con máximo rendimiento...")
        print("📊 Configuración: 5000 episodios finales, 500 evaluaciones, DescentEnv real")
        
        # Ejecutar experimento FLAN directamente
        result = subprocess.run([
            python_exe, 'flan_qlearning_solution.py'
        ], capture_output=False, text=True, check=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ FLAN completado en {duration:.1f} segundos ({duration/60:.1f} minutos)")
        
        # Verificar archivos generados
        generated_files = []
        expected_files = [
            'flan_results.json',
            'flan_results.png',
            'models_fina/',
            'models_media/',
            'models_gruesa/'
        ]
        
        for file_path in expected_files:
            if os.path.exists(file_path):
                generated_files.append(file_path)
                print(f"  ✅ Generado: {file_path}")
            else:
                print(f"  ⚠️  No encontrado: {file_path}")
        
        # Volver al directorio raíz
        os.chdir(original_dir)
        
        return True, generated_files
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error en FLAN: {e}")
        if os.getcwd() != original_dir:
            os.chdir(original_dir)
        return False, []
    except Exception as e:
        print(f"\n❌ Error inesperado en FLAN: {e}")
        if os.getcwd() != original_dir:
            os.chdir(original_dir)
        return False, []

def run_bored_experiment(python_exe):
    """Ejecuta el experimento BORED con paralelización completa"""
    print("="*80)
    print("EJECUTANDO PROYECTO BORED con Paralelización Completa")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Cambiar al directorio de BORED
        original_dir = os.getcwd()
        os.chdir('tactix')
        
        print("🚀 Iniciando experimento BORED con máximo rendimiento...")
        print("🎮 Configuración: Minimax vs Expectimax, Alpha-Beta Pruning, Múltiples heurísticas")
        
        # Ejecutar experimento BORED directamente
        result = subprocess.run([
            python_exe, 'bored_solution.py'
        ], capture_output=False, text=True, check=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ BORED completado en {duration:.1f} segundos ({duration/60:.1f} minutos)")
        
        # Verificar archivos generados
        generated_files = []
        expected_files = [
            'bored_results.json',
            'bored_results.png',
            'bored_models.pkl'
        ]
        
        for file_path in expected_files:
            if os.path.exists(file_path):
                generated_files.append(file_path)
                print(f"  ✅ Generado: {file_path}")
            else:
                print(f"  ⚠️  No encontrado: {file_path}")
        
        # Volver al directorio raíz
        os.chdir(original_dir)
        
        return True, generated_files
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error en BORED: {e}")
        if os.getcwd() != original_dir:
            os.chdir(original_dir)
        return False, []
    except Exception as e:
        print(f"\n❌ Error inesperado en BORED: {e}")
        if os.getcwd() != original_dir:
            os.chdir(original_dir)
        return False, []

def run_demo_tests(python_exe):
    """Ejecuta demos rápidos para verificar que todo funciona"""
    print("="*80)
    print("EJECUTANDO DEMOS DE VERIFICACIÓN")
    print("="*80)
    
    demos_successful = 0
    total_demos = 2
    
    # Demo FLAN
    try:
        print("🧪 Probando demo FLAN...")
        original_dir = os.getcwd()
        os.chdir('descent-env')
        
        result = subprocess.run([
            python_exe, 'demo_flan.py'
        ], capture_output=True, text=True, check=True, timeout=300)  # 5 minutos máximo
        
        print("✅ Demo FLAN exitoso")
        demos_successful += 1
        os.chdir(original_dir)
        
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"❌ Demo FLAN falló: {e}")
        if os.getcwd() != original_dir:
            os.chdir(original_dir)
    
    # Demo BORED
    try:
        print("🧪 Probando demo BORED...")
        original_dir = os.getcwd()
        os.chdir('tactix')
        
        result = subprocess.run([
            python_exe, 'demo_bored.py'
        ], capture_output=True, text=True, check=True, timeout=300)  # 5 minutos máximo
        
        print("✅ Demo BORED exitoso")
        demos_successful += 1
        os.chdir(original_dir)
        
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"❌ Demo BORED falló: {e}")
        if os.getcwd() != original_dir:
            os.chdir(original_dir)
    
    return demos_successful == total_demos

def generate_final_report(flan_files, bored_files):
    """Genera un reporte final consolidado"""
    print("="*80)
    print("GENERANDO REPORTE FINAL CONSOLIDADO")
    print("="*80)
    
    print("📁 Archivos generados:")
    print()
    
    print("🔬 PROYECTO FLAN (Q-Learning para Control de Descenso):")
    if flan_files:
        for file_path in flan_files:
            print(f"  ✅ descent-env/{file_path}")
    else:
        print("  ❌ No se generaron archivos")
    
    print()
    print("🎮 PROYECTO BORED (Minimax/Expectimax para TacTix):")
    if bored_files:
        for file_path in bored_files:
            print(f"  ✅ tactix/{file_path}")
    else:
        print("  ❌ No se generaron archivos")
    
    print()
    print("📊 CONFIGURACIÓN UTILIZADA:")
    print("  🚀 DescentEnv REAL (BlueSky) para máximo rendimiento")
    print("  ⚡ Paralelización completa con todos los cores de CPU")
    print("  📈 Configuración competitiva: 5000 episodios de entrenamiento")
    print("  🎯 Evaluación robusta: 500 episodios de evaluación")
    print("  🧠 Técnicas avanzadas: Double Q-Learning, Reward Shaping, Alpha-Beta Pruning")

def main():
    """Función principal"""
    total_start_time = time.time()
    
    print("🚀 EJECUTOR MAESTRO - PROYECTOS FLAN Y BORED")
    print("Optimizado para MacBook M1 Pro con DescentEnv Real")
    print("="*80)
    
    # Verificar entorno virtual
    python_exe = check_virtual_environment()
    if not python_exe:
        return 1
    
    # Configurar entorno
    setup_environment()
    
    # Ejecutar demos primero para verificar
    print("🧪 Verificando configuración con demos...")
    demo_success = run_demo_tests(python_exe)
    
    if not demo_success:
        print("⚠️  Los demos fallaron. Verifique la configuración antes de continuar.")
        response = input("¿Continuar con experimentos completos? (y/N): ")
        if response.lower() != 'y':
            return 1
    
    # Ejecutar experimentos completos
    flan_success, flan_files = run_flan_experiment(python_exe)
    bored_success, bored_files = run_bored_experiment(python_exe)
    
    # Generar reporte final
    generate_final_report(flan_files, bored_files)
    
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
        print("🚀 Experimentos ejecutados con DescentEnv REAL para máxima calidad")
        return 0
    else:
        print("\n⚠️  Algunos experimentos fallaron. Revisar logs arriba.")
        print("💡 Tip: Ejecute los demos individuales para diagnosticar problemas:")
        print("   - cd descent-env && python3 demo_flan.py")
        print("   - cd tactix && python3 demo_bored.py")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n🛑 Experimento interrumpido por el usuario")
        print("💾 Los archivos parciales se han guardado")
        sys.exit(130) 