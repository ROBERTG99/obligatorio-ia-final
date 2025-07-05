#!/usr/bin/env python3
"""
🚀 VERIFICACIÓN DE MÁXIMO RENDIMIENTO
Script para verificar y optimizar el uso de TODO el procesamiento disponible
"""

import psutil
import os
import multiprocessing as mp
import time
from datetime import datetime
import json

def verificar_recursos_detallado():
    """Verificación detallada de recursos del sistema"""
    print("🖥️  VERIFICACIÓN DETALLADA DE RECURSOS")
    print("="*70)
    
    # CPU
    cpu_logical = psutil.cpu_count(logical=True)
    cpu_physical = psutil.cpu_count(logical=False)
    cpu_freq = psutil.cpu_freq()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    print(f"🔧 CPU:")
    print(f"   • Cores Lógicos: {cpu_logical}")
    print(f"   • Cores Físicos: {cpu_physical}")
    print(f"   • Frecuencia Max: {cpu_freq.max if cpu_freq else 'N/A'} MHz")
    print(f"   • Utilización Actual: {cpu_percent}%")
    
    # RAM
    ram = psutil.virtual_memory()
    print(f"\n💾 RAM:")
    print(f"   • Total: {ram.total / (1024**3):.1f} GB")
    print(f"   • Disponible: {ram.available / (1024**3):.1f} GB")
    print(f"   • Utilización: {ram.percent}%")
    print(f"   • Libre: {ram.free / (1024**3):.1f} GB")
    
    # Procesos
    print(f"\n🔄 PROCESOS:")
    print(f"   • Multiprocessing CPU Count: {mp.cpu_count()}")
    print(f"   • Procesos activos: {len(psutil.pids())}")
    
    # Carga del sistema
    if hasattr(os, 'getloadavg'):
        load_avg = os.getloadavg()
        print(f"   • Carga promedio: {load_avg[0]:.1f} (1min), {load_avg[1]:.1f} (5min), {load_avg[2]:.1f} (15min)")
    
    return {
        'cpu_logical': cpu_logical,
        'cpu_physical': cpu_physical,
        'cpu_freq_max': cpu_freq.max if cpu_freq else 3000,
        'cpu_percent': cpu_percent,
        'ram_total_gb': ram.total / (1024**3),
        'ram_available_gb': ram.available / (1024**3),
        'ram_percent': ram.percent,
        'mp_cpu_count': mp.cpu_count()
    }

def calcular_configuracion_optima(recursos):
    """Calcula la configuración óptima para máximo rendimiento"""
    print("\n🎯 CALCULANDO CONFIGURACIÓN ÓPTIMA")
    print("="*50)
    
    # Configuración base
    cpu_logical = recursos['cpu_logical']
    ram_available = recursos['ram_available_gb']
    cpu_percent = recursos['cpu_percent']
    
    # Calcular workers óptimos
    if cpu_percent < 50:
        # Sistema poco cargado: usar TODOS los cores
        workers_optimos = cpu_logical
        print(f"   🚀 Sistema poco cargado ({cpu_percent}%)")
        print(f"   💪 USANDO TODOS LOS CORES: {workers_optimos}")
    elif cpu_percent < 80:
        # Sistema moderadamente cargado: usar 80% cores
        workers_optimos = max(1, int(cpu_logical * 0.8))
        print(f"   ⚡ Sistema moderadamente cargado ({cpu_percent}%)")
        print(f"   💪 USANDO 80% DE CORES: {workers_optimos}")
    else:
        # Sistema muy cargado: usar 50% cores
        workers_optimos = max(1, int(cpu_logical * 0.5))
        print(f"   ⚠️  Sistema muy cargado ({cpu_percent}%)")
        print(f"   🐌 USANDO 50% DE CORES: {workers_optimos}")
    
    # Calcular RAM por proceso
    ram_por_proceso = ram_available / workers_optimos
    print(f"   📊 RAM por proceso: {ram_por_proceso:.1f} GB")
    
    # Calcular batch sizes óptimos
    if ram_por_proceso > 1.0:
        batch_size_grande = min(workers_optimos, 32)
        batch_size_medio = min(workers_optimos, 16)
        print(f"   🚀 RAM abundante: Batch grande={batch_size_grande}, medio={batch_size_medio}")
    elif ram_por_proceso > 0.5:
        batch_size_grande = min(workers_optimos, 16)
        batch_size_medio = min(workers_optimos, 8)
        print(f"   ⚡ RAM suficiente: Batch grande={batch_size_grande}, medio={batch_size_medio}")
    else:
        batch_size_grande = min(workers_optimos, 8)
        batch_size_medio = min(workers_optimos, 4)
        print(f"   ⚠️  RAM limitada: Batch grande={batch_size_grande}, medio={batch_size_medio}")
    
    # Configuración de episodios escalables
    if ram_por_proceso > 1.0 and cpu_logical >= 8:
        episodios_config = {
            'rapido': 500,
            'medio': 2000,
            'intensivo': 8000
        }
        print(f"   🏆 Sistema ULTRA POTENTE: Episodios {episodios_config}")
    elif ram_por_proceso > 0.5 and cpu_logical >= 4:
        episodios_config = {
            'rapido': 200,
            'medio': 1000,
            'intensivo': 4000
        }
        print(f"   💪 Sistema POTENTE: Episodios {episodios_config}")
    else:
        episodios_config = {
            'rapido': 100,
            'medio': 500,
            'intensivo': 2000
        }
        print(f"   🐌 Sistema ESTÁNDAR: Episodios {episodios_config}")
    
    configuracion_optima = {
        'workers_optimos': workers_optimos,
        'batch_size_grande': batch_size_grande,
        'batch_size_medio': batch_size_medio,
        'ram_por_proceso': ram_por_proceso,
        'episodios': episodios_config,
        'nivel_sistema': 'ULTRA' if ram_por_proceso > 1.0 and cpu_logical >= 8 
                        else 'POTENTE' if ram_por_proceso > 0.5 and cpu_logical >= 4 
                        else 'ESTÁNDAR'
    }
    
    return configuracion_optima

def generar_comando_optimizado(configuracion):
    """Genera comandos optimizados para ejecutar"""
    print(f"\n📋 COMANDOS OPTIMIZADOS PARA TU SISTEMA")
    print("="*60)
    
    nivel = configuracion['nivel_sistema']
    workers = configuracion['workers_optimos']
    
    print(f"🎯 CONFIGURACIÓN: {nivel}")
    print(f"   • Workers: {workers}")
    print(f"   • Batch grande: {configuracion['batch_size_grande']}")
    print(f"   • RAM por proceso: {configuracion['ram_por_proceso']:.1f} GB")
    
    print(f"\n🚀 COMANDOS RECOMENDADOS:")
    
    # Comando 1: Optimización continua
    print(f"\n1. OPTIMIZACIÓN CONTINUA CON MÁXIMO RENDIMIENTO:")
    print(f"   python ejecutar_ultra_performance.py")
    print(f"   (Usará automáticamente {workers} workers)")
    
    # Comando 2: Experimento express
    print(f"\n2. EXPERIMENTO EXPRESS OPTIMIZADO:")
    print(f"   python ejecutar_experimento_express.py")
    print(f"   (Configuración automática para {nivel})")
    
    # Comando 3: Experimento completo
    print(f"\n3. EXPERIMENTO COMPLETO (MÁXIMO RENDIMIENTO):")
    print(f"   python flan_qlearning_solution.py")
    print(f"   (Paralelización masiva con {workers} workers)")
    
    # Comando 4: Versión segura
    print(f"\n4. VERSIÓN SEGURA (Si hay problemas):")
    print(f"   python flan_qlearning_solution.py safe")
    print(f"   (Usa configuración conservadora)")

def test_paralelizacion_simple():
    """Test simple para verificar que la paralelización funciona"""
    print(f"\n🧪 TEST DE PARALELIZACIÓN")
    print("="*40)
    
    def trabajo_simple(x):
        """Trabajo simple para test"""
        time.sleep(0.1)
        return x * x
    
    # Test secuencial
    print("   🐌 Test secuencial...")
    start_time = time.time()
    resultados_seq = [trabajo_simple(i) for i in range(10)]
    tiempo_secuencial = time.time() - start_time
    
    # Test paralelo
    print("   🚀 Test paralelo...")
    start_time = time.time()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        resultados_par = pool.map(trabajo_simple, range(10))
    tiempo_paralelo = time.time() - start_time
    
    # Resultados
    aceleracion = tiempo_secuencial / tiempo_paralelo if tiempo_paralelo > 0 else 0
    print(f"   ✅ Tiempo secuencial: {tiempo_secuencial:.2f}s")
    print(f"   ✅ Tiempo paralelo: {tiempo_paralelo:.2f}s")
    print(f"   🚀 Aceleración: {aceleracion:.1f}x")
    
    if aceleracion > 2.0:
        print(f"   🏆 PARALELIZACIÓN FUNCIONANDO PERFECTAMENTE")
    elif aceleracion > 1.5:
        print(f"   ✅ Paralelización funcionando bien")
    else:
        print(f"   ⚠️  Paralelización limitada")
    
    return aceleracion

def guardar_configuracion_optimizada(recursos, configuracion):
    """Guarda la configuración optimizada para uso futuro"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    config_completa = {
        'timestamp': timestamp,
        'recursos_sistema': recursos,
        'configuracion_optimizada': configuracion,
        'comandos_recomendados': {
            'ultra_performance': 'python ejecutar_ultra_performance.py',
            'experimento_express': 'python ejecutar_experimento_express.py',
            'experimento_completo': 'python flan_qlearning_solution.py',
            'version_segura': 'python flan_qlearning_solution.py safe'
        }
    }
    
    filename = f"configuracion_max_performance_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(config_completa, f, indent=2)
    
    print(f"\n💾 CONFIGURACIÓN GUARDADA EN: {filename}")
    return filename

def main():
    """Función principal"""
    print("🚀 VERIFICACIÓN DE MÁXIMO RENDIMIENTO")
    print("="*80)
    print("🎯 OBJETIVO: Verificar y optimizar el uso de TODO el procesamiento")
    print("="*80)
    
    # Verificar recursos
    recursos = verificar_recursos_detallado()
    
    # Calcular configuración óptima
    configuracion = calcular_configuracion_optima(recursos)
    
    # Generar comandos optimizados
    generar_comando_optimizado(configuracion)
    
    # Test de paralelización
    aceleracion = test_paralelizacion_simple()
    
    # Guardar configuración
    config_file = guardar_configuracion_optimizada(recursos, configuracion)
    
    # Resumen final
    print(f"\n✅ VERIFICACIÓN COMPLETADA")
    print("="*50)
    print(f"   • Nivel del sistema: {configuracion['nivel_sistema']}")
    print(f"   • Workers óptimos: {configuracion['workers_optimos']}")
    print(f"   • Aceleración paralela: {aceleracion:.1f}x")
    print(f"   • Configuración guardada: {config_file}")
    
    if configuracion['nivel_sistema'] == 'ULTRA':
        print(f"\n🏆 SISTEMA ULTRA POTENTE DETECTADO")
        print(f"   • Listo para paralelización masiva")
        print(f"   • Usar: python ejecutar_ultra_performance.py")
    elif configuracion['nivel_sistema'] == 'POTENTE':
        print(f"\n💪 SISTEMA POTENTE DETECTADO")
        print(f"   • Listo para paralelización completa")
        print(f"   • Usar: python ejecutar_experimento_express.py")
    else:
        print(f"\n⚡ SISTEMA ESTÁNDAR DETECTADO")
        print(f"   • Usar configuración conservadora")
        print(f"   • Usar: python flan_qlearning_solution.py safe")

if __name__ == "__main__":
    main() 