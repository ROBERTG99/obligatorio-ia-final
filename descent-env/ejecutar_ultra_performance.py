#!/usr/bin/env python3
"""
🚀 EJECUTOR ULTRA PERFORMANCE
Script simple para ejecutar optimización con máximo rendimiento
"""

import time
import psutil
from datetime import datetime

def mostrar_recursos_sistema():
    """Muestra los recursos del sistema"""
    cpu_logical = psutil.cpu_count(logical=True)
    cpu_physical = psutil.cpu_count(logical=False)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    ram_available_gb = psutil.virtual_memory().available / (1024**3)
    
    print("🖥️  RECURSOS DEL SISTEMA")
    print("="*50)
    print(f"   • CPU Lógicos: {cpu_logical} cores")
    print(f"   • CPU Físicos: {cpu_physical} cores")
    print(f"   • RAM Total: {ram_gb:.1f} GB")
    print(f"   • RAM Disponible: {ram_available_gb:.1f} GB")
    print(f"   • Utilización CPU: {psutil.cpu_percent(interval=1):.1f}%")
    print(f"   • Utilización RAM: {psutil.virtual_memory().percent:.1f}%")
    
    return {
        'cpu_logical': cpu_logical,
        'cpu_physical': cpu_physical,
        'ram_available_gb': ram_available_gb
    }

def ejecutar_optimizacion_continua_ultra():
    """Ejecuta optimización continua con retroactividad usando máximo rendimiento"""
    print("🚀 OPTIMIZACIÓN CONTINUA ULTRA PERFORMANCE")
    print("="*80)
    print("🎯 OBJETIVO: Usar TODO el procesamiento + análisis retroactivo")
    print("💪 ESTRATEGIA: Episodios escalables + paralelización masiva")
    print("="*80)
    
    # Mostrar recursos
    recursos = mostrar_recursos_sistema()
    
    # Configuración del usuario
    print(f"\n⚙️  CONFIGURACIÓN")
    objetivo_mejora = float(input("📊 Mejora objetivo (puntos): ") or "10")
    
    # Estimar capacidad
    capacidad_estimada = recursos['cpu_logical'] * recursos['ram_available_gb'] / 4
    print(f"\n📊 CAPACIDAD ESTIMADA DEL SISTEMA:")
    print(f"   • Configuraciones paralelas: ~{recursos['cpu_logical']}")
    print(f"   • RAM por proceso: ~{recursos['ram_available_gb']/recursos['cpu_logical']:.1f} GB")
    print(f"   • Capacidad total: {capacidad_estimada:.0f} puntos")
    
    if capacidad_estimada < 8:
        print(f"   ⚠️  Sistema limitado. Usar configuración conservadora.")
        usar_ultra = False
    else:
        print(f"   🚀 Sistema potente. Usar configuración ULTRA.")
        usar_ultra = True
    
    # Confirmación
    print(f"\n🚨 ADVERTENCIA:")
    if usar_ultra:
        print(f"   • Se usarán {recursos['cpu_logical']} cores simultáneos")
        print(f"   • Esto puede sobrecargar el sistema temporalmente")
        print(f"   • El sistema puede volverse lento durante la optimización")
    else:
        print(f"   • Se usará configuración conservadora")
        print(f"   • {min(recursos['cpu_logical'], 4)} cores máximo")
    
    confirm = input("\n¿Proceder con optimización ULTRA PERFORMANCE? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ Optimización cancelada")
        return
    
    print(f"\n🔥 INICIANDO OPTIMIZACIÓN ULTRA PERFORMANCE...")
    print(f"   • Hora inicio: {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        # Importar y ejecutar optimizador
        from optimizacion_continua_retroactiva import OptimizadorContinuoRetroactivo
        
        # Crear optimizador con configuración ultra
        optimizador = OptimizadorContinuoRetroactivo(
            objetivo_mejora=objetivo_mejora,
            max_iteraciones=20 if usar_ultra else 10
        )
        
        # Ajustar configuración para ultra performance
        if usar_ultra:
            optimizador.episodios_base = 1000      # Más episodios base
            optimizador.episodios_maximos = 10000  # Máximo muy alto
            optimizador.factor_escalado = 2.5      # Escalado más agresivo
        
        # Ejecutar optimización
        start_time = time.time()
        resultado = optimizador.ejecutar_optimizacion_continua()
        end_time = time.time()
        
        # Mostrar resultados
        print(f"\n✅ OPTIMIZACIÓN COMPLETADA")
        print(f"   • Tiempo total: {(end_time - start_time)/60:.1f} minutos")
        
        if resultado:
            mejora_conseguida = resultado['score'] - optimizador.baseline_score
            print(f"   • Score final: {resultado['score']:.2f}")
            print(f"   • Mejora conseguida: +{mejora_conseguida:.2f}")
            print(f"   • Objetivo era: +{objetivo_mejora:.2f}")
            
            if mejora_conseguida >= objetivo_mejora:
                print(f"   🏆 ¡OBJETIVO ALCANZADO!")
            else:
                print(f"   ⚠️  Objetivo no alcanzado. Falta: +{objetivo_mejora - mejora_conseguida:.2f}")
            
            print(f"   • Configuración exitosa: {resultado['params']}")
        else:
            print(f"   ❌ No se obtuvo resultado válido")
            
    except ImportError as e:
        print(f"❌ Error importando optimizador: {e}")
        print("💡 Ejecutando versión simplificada...")
        ejecutar_version_simplificada(objetivo_mejora, usar_ultra)
    except Exception as e:
        print(f"❌ Error en optimización: {e}")
        import traceback
        traceback.print_exc()

def ejecutar_version_simplificada(objetivo_mejora: float, usar_ultra: bool):
    """Versión simplificada que usa el grid search existente"""
    try:
        from flan_qlearning_solution import (
            DescentEnv, MockDescentEnv, DiscretizationScheme,
            QLearningAgent, HyperparameterOptimizer, BLUESKY_AVAILABLE, MOCK_AVAILABLE
        )
        from contextlib import redirect_stdout, redirect_stderr
        import os
        
        print("\n🔥 EJECUTANDO VERSIÓN SIMPLIFICADA CON MÁXIMO RENDIMIENTO")
        
        # Crear entorno
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                if BLUESKY_AVAILABLE:
                    env = DescentEnv(render_mode=None)
                    print("   ✅ Usando DescentEnv REAL")
                elif MOCK_AVAILABLE:
                    env = MockDescentEnv(render_mode=None)
                    print("   ✅ Usando MockDescentEnv")
                else:
                    raise ImportError("No hay entornos disponibles")
        
        # Discretización optimizada
        discretization = DiscretizationScheme("UltraPerf", 30, 25, 25, 25, 15)
        
        # Grid de hiperparámetros ampliado para máximo rendimiento
        if usar_ultra:
            param_grid = {
                'learning_rate': [0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
                'discount_factor': [0.99, 0.995, 0.999, 0.9999],
                'epsilon': [0.3, 0.5, 0.7, 0.9, 0.95],
                'use_double_q': [True, False],
                'use_reward_shaping': [True]
            }
        else:
            param_grid = {
                'learning_rate': [0.3, 0.5, 0.7],
                'discount_factor': [0.99, 0.999],
                'epsilon': [0.5, 0.7, 0.9],
                'use_double_q': [True],
                'use_reward_shaping': [True]
            }
        
        # Ejecutar optimización
        optimizer = HyperparameterOptimizer(env, discretization)
        result = optimizer.grid_search('qlearning', param_grid)
        
        print(f"\n✅ OPTIMIZACIÓN SIMPLIFICADA COMPLETADA")
        print(f"   • Mejor score: {result['best_score']:.2f}")
        print(f"   • Mejores parámetros: {result['best_params']}")
        
        env.close()
        
    except Exception as e:
        print(f"❌ Error en versión simplificada: {e}")

def main():
    """Función principal"""
    try:
        ejecutar_optimizacion_continua_ultra()
    except KeyboardInterrupt:
        print("\n⚠️  Optimización interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 