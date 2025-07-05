#!/usr/bin/env python3
"""
🔍 VERIFICACIÓN RÁPIDA FLAN
Script para verificar que todo funciona antes del experimento completo
"""

import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
import os

def test_bluesky_import():
    """Prueba 1: Verificar importación de BlueSky"""
    print("🔍 PRUEBA 1: Importación BlueSky")
    print("-" * 40)
    
    try:
        from descent_env import DescentEnv
        print("✅ DescentEnv importado correctamente")
        return True
    except ImportError as e:
        print(f"❌ Error importando DescentEnv: {e}")
        return False

def test_environment_creation():
    """Prueba 2: Crear y probar entorno"""
    print("\n🔍 PRUEBA 2: Creación de entorno")
    print("-" * 40)
    
    try:
        from descent_env import DescentEnv
        
        # Suprimir warnings de BlueSky
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                env = DescentEnv(render_mode=None)
        
        print("✅ Entorno DescentEnv creado exitosamente")
        
        # Probar reset
        obs, info = env.reset()
        print("✅ Reset del entorno funcional")
        print(f"   Observación keys: {list(obs.keys())}")
        print(f"   Info: {info}")
        
        # Probar un paso
        import numpy as np
        action = np.array([0.1])
        obs, reward, done, truncated, info = env.step(action)
        print("✅ Step del entorno funcional")
        print(f"   Reward: {reward:.3f}")
        print(f"   Done: {done}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Error en entorno: {e}")
        traceback.print_exc()
        return False

def test_agent_creation():
    """Prueba 3: Crear agentes con mejoras automáticas"""
    print("\n🔍 PRUEBA 3: Creación de agentes")
    print("-" * 40)
    
    try:
        from flan_qlearning_solution import (
            DiscretizationScheme, 
            QLearningAgent, 
            StochasticQLearningAgent,
            CONFIG_AUTOMATICO,
            RewardShaperAutomatico
        )
        
        # Crear discretización
        discretization = DiscretizationScheme("Test", 10, 10, 10, 10, 5)
        print("✅ Discretización creada")
        
        # Crear agente Q-Learning con mejoras automáticas
        agent_ql = QLearningAgent(discretization)
        print("✅ QLearningAgent creado")
        print(f"   Learning rate: {agent_ql.learning_rate}")
        print(f"   Epsilon: {agent_ql.epsilon}")
        print(f"   Reward shaper: {type(agent_ql.reward_shaper).__name__}")
        
        # Crear agente Stochastic
        agent_stoch = StochasticQLearningAgent(discretization)
        print("✅ StochasticQLearningAgent creado")
        print(f"   Reward shaper: {type(agent_stoch.reward_shaper).__name__}")
        
        # Verificar CONFIG_AUTOMATICO
        print("✅ CONFIG_AUTOMATICO verificado:")
        for key, value in CONFIG_AUTOMATICO.items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creando agentes: {e}")
        traceback.print_exc()
        return False

def test_training_mini():
    """Prueba 4: Entrenamiento mini (10 episodios)"""
    print("\n🔍 PRUEBA 4: Entrenamiento mini")
    print("-" * 40)
    
    try:
        from descent_env import DescentEnv
        from flan_qlearning_solution import (
            DiscretizationScheme, 
            QLearningAgent, 
            QLearningTrainer
        )
        
        # Crear componentes
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                env = DescentEnv(render_mode=None)
        
        discretization = DiscretizationScheme("MiniTest", 5, 5, 5, 5, 3)
        agent = QLearningAgent(discretization)
        trainer = QLearningTrainer(env, agent, discretization)
        
        print("🏃 Iniciando entrenamiento mini (10 episodios)...")
        start_time = time.time()
        
        rewards = trainer.train(episodes=10, verbose=False)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print("✅ Entrenamiento mini completado")
        print(f"   Tiempo: {training_time:.2f} segundos")
        print(f"   Recompensa final: {rewards[-1]:.3f}")
        print(f"   Tiempo por episodio: {training_time/10:.3f} segundos")
        
        # Extrapolar a experimento completo
        total_episodes = 256000
        estimated_hours = (training_time / 10) * total_episodes / 3600
        
        print(f"\n📊 EXTRAPOLACIÓN AL EXPERIMENTO COMPLETO:")
        print(f"   Episodios totales: {total_episodes:,}")
        print(f"   Tiempo estimado: {estimated_hours:.1f} horas")
        
        env.close()
        return True, estimated_hours
        
    except Exception as e:
        print(f"❌ Error en entrenamiento mini: {e}")
        traceback.print_exc()
        return False, None

def main():
    """Función principal de verificación"""
    print("🎯 VERIFICACIÓN RÁPIDA FLAN")
    print("=" * 80)
    print("🔥 Probando configuración antes del experimento completo")
    print("⏱️ Duración estimada: 2-3 minutos")
    print()
    
    tests_passed = 0
    total_tests = 4
    estimated_time = None
    
    # Ejecutar pruebas
    if test_bluesky_import():
        tests_passed += 1
    
    if test_environment_creation():
        tests_passed += 1
    
    if test_agent_creation():
        tests_passed += 1
    
    success, est_time = test_training_mini()
    if success:
        tests_passed += 1
        estimated_time = est_time
    
    # Reporte final
    print("\n" + "=" * 80)
    print("📊 REPORTE DE VERIFICACIÓN")
    print("=" * 80)
    print(f"✅ Pruebas pasadas: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 ¡TODAS LAS PRUEBAS PASARON!")
        print("🚀 Sistema listo para el experimento completo")
        
        if estimated_time:
            print(f"\n⏱️ TIEMPO ESTIMADO ACTUALIZADO: {estimated_time:.1f} horas")
            
            # Sugerencia basada en tiempo
            if estimated_time > 40:
                print("⚠️ ADVERTENCIA: Tiempo muy alto (>40h)")
                print("   Considera usar menos episodios para prueba inicial")
            elif estimated_time > 30:
                print("✅ Tiempo dentro de lo esperado (30-40h)")
            else:
                print("🚀 Tiempo mejor de lo esperado (<30h)")
        
        print("\n🎯 SIGUIENTE PASO:")
        print("   python3 ejecutar_experimento_completo.py")
        
    else:
        print("❌ ALGUNAS PRUEBAS FALLARON")
        print("🔧 Revisa los errores antes de continuar")
        
        if tests_passed == 0:
            print("💡 Sugerencia: Verifica instalación de BlueSky")
        elif tests_passed < 3:
            print("💡 Sugerencia: Problemas de configuración básica")
        else:
            print("💡 Sugerencia: Problema menor, podrías continuar")

if __name__ == "__main__":
    main() 