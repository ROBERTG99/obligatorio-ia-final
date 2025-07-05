#!/usr/bin/env python3
"""
🚀 EXPERIMENTO EXPRESS FLAN - VERSIÓN OPTIMIZADA PARA TIEMPO
Experimento completo pero con episodios reducidos para validación rápida
"""

import os
import time
import sys
import json
import psutil
from datetime import datetime, timedelta
from contextlib import redirect_stdout, redirect_stderr

def estimate_express_time():
    """Estima tiempo del experimento express"""
    print("⏱️ EXPERIMENTO EXPRESS - ESTIMACIÓN DE TIEMPO")
    print("=" * 60)
    
    # Configuración express
    hyperparams_episodes = 1000    # Era 6,000
    final_training = 5000         # Era 60,000  
    final_evaluation = 500        # Era 8,000
    
    total_episodes = (
        4 * hyperparams_episodes +  # Q-Learning hyperparams
        8 * hyperparams_episodes +  # Stochastic hyperparams
        2 * final_training +        # Entrenamiento final
        2 * final_evaluation        # Evaluación final
    )
    
    # Basado en datos reales: 1.699 segundos por episodio
    time_per_episode = 1.7  # segundos
    total_time_hours = (total_episodes * time_per_episode) / 3600
    
    print(f"📊 CONFIGURACIÓN EXPRESS:")
    print(f"   • Hiperparámetros: {hyperparams_episodes:,} episodios por combinación")
    print(f"   • Entrenamiento final: {final_training:,} episodios por agente")
    print(f"   • Evaluación: {final_evaluation:,} episodios por agente")
    print(f"   • TOTAL: {total_episodes:,} episodios")
    print()
    print(f"⏰ TIEMPO ESTIMADO:")
    print(f"   • Tiempo por episodio: {time_per_episode} segundos")
    print(f"   • Tiempo total: {total_time_hours:.1f} horas")
    
    start_time = datetime.now()
    estimated_end = start_time + timedelta(hours=total_time_hours)
    
    print(f"   • Inicio: {start_time.strftime('%H:%M:%S')}")
    print(f"   • Finalización: {estimated_end.strftime('%H:%M:%S')}")
    print(f"   • Duración: {estimated_end - start_time}")
    print()
    
    return total_time_hours

def run_express_experiment():
    """Ejecuta el experimento express"""
    print("🚀 INICIANDO EXPERIMENTO EXPRESS")
    print("=" * 80)
    
    # Verificar DescentEnv REAL
    try:
        from descent_env import DescentEnv
        print("✅ DescentEnv REAL confirmado")
    except ImportError:
        print("❌ ERROR: DescentEnv no disponible")
        return False
    
    print("\n🏃 Ejecutando experimento express...")
    start_time = time.time()
    
    try:
        # Importar componentes principales
        from flan_qlearning_solution import (
            DescentEnv, DiscretizationScheme, QLearningAgent, 
            StochasticQLearningAgent, QLearningTrainer,
            PerformanceEvaluator, HyperparameterOptimizer,
            CONFIG_AUTOMATICO
        )
        
        print("🎯 Creando entorno y configuración...")
        
        # Crear entorno (suprimir warnings)
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                env = DescentEnv(render_mode=None)
        
        # Esquema de discretización optimizado (pero más pequeño)
        discretization = DiscretizationScheme("Express", 20, 15, 15, 15, 10)
        
        # CONFIGURACIÓN EXPRESS
        EXPRESS_CONFIG = {
            'hyperparams_episodes': 1000,    # vs 6,000 original
            'final_training': 5000,          # vs 60,000 original  
            'final_evaluation': 500,         # vs 8,000 original
        }
        
        print(f"📊 Configuración Express: {EXPRESS_CONFIG}")
        
        results_summary = {}
        
        # === OPTIMIZACIÓN Q-LEARNING ===
        print("\n1. 🔍 Optimizando Q-Learning (Express)...")
        
        # Grid search con menos combinaciones y episodios
        param_grid_ql = {
            'learning_rate': [0.8, 0.95],      # vs [0.5, 0.7] original
            'discount_factor': [0.999],
            'epsilon': [0.8, 0.95],            # vs [0.7, 0.8] original
            'use_double_q': [True],
            'use_reward_shaping': [True]
        }
        
        # Crear optimizador express
        optimizer = HyperparameterOptimizer(env, discretization)
        
        # Modificar temporalmente el train_and_evaluate_agent para express
        original_training = 6000
        optimizer.express_episodes = EXPRESS_CONFIG['hyperparams_episodes']
        
        # Ejecutar grid search
        qlearning_results = optimizer.grid_search('qlearning', param_grid_ql)
        print(f"✅ Mejores parámetros Q-Learning: {qlearning_results['best_params']}")
        
        # === ENTRENAMIENTO FINAL Q-LEARNING ===
        print(f"\n2. 🏋️ Entrenando Q-Learning final ({EXPRESS_CONFIG['final_training']} episodios)...")
        
        best_qlearning_agent = QLearningAgent(discretization, **qlearning_results['best_params'])
        qlearning_trainer = QLearningTrainer(env, best_qlearning_agent, discretization)
        qlearning_trainer.train(episodes=EXPRESS_CONFIG['final_training'], verbose=True)
        
        # === EVALUACIÓN Q-LEARNING ===
        print(f"\n3. 📊 Evaluando Q-Learning ({EXPRESS_CONFIG['final_evaluation']} episodios)...")
        
        qlearning_evaluator = PerformanceEvaluator(env, best_qlearning_agent, discretization)
        qlearning_eval = qlearning_evaluator.evaluate_multiple_episodes(
            num_episodes=EXPRESS_CONFIG['final_evaluation']
        )
        
        # === OPTIMIZACIÓN STOCHASTIC ===
        print("\n4. 🔍 Optimizando Stochastic Q-Learning (Express)...")
        
        param_grid_stoch = {
            'learning_rate': [0.6, 0.8],
            'discount_factor': [0.999],
            'epsilon': [0.8, 0.9],
            'sample_size': [10, 12],
            'use_reward_shaping': [True]
        }
        
        stochastic_results = optimizer.grid_search('stochastic', param_grid_stoch)
        print(f"✅ Mejores parámetros Stochastic: {stochastic_results['best_params']}")
        
        # === ENTRENAMIENTO FINAL STOCHASTIC ===
        print(f"\n5. 🏋️ Entrenando Stochastic final ({EXPRESS_CONFIG['final_training']} episodios)...")
        
        best_stochastic_agent = StochasticQLearningAgent(discretization, **stochastic_results['best_params'])
        stochastic_trainer = QLearningTrainer(env, best_stochastic_agent, discretization)
        stochastic_trainer.train(episodes=EXPRESS_CONFIG['final_training'], verbose=True)
        
        # === EVALUACIÓN STOCHASTIC ===
        print(f"\n6. 📊 Evaluando Stochastic ({EXPRESS_CONFIG['final_evaluation']} episodios)...")
        
        stochastic_evaluator = PerformanceEvaluator(env, best_stochastic_agent, discretization)
        stochastic_eval = stochastic_evaluator.evaluate_multiple_episodes(
            num_episodes=EXPRESS_CONFIG['final_evaluation']
        )
        
        # === GUARDAR RESULTADOS ===
        print("\n7. 💾 Guardando resultados...")
        
        # Crear directorio de modelos
        models_dir = "models_express_flan"
        os.makedirs(models_dir, exist_ok=True)
        
        # Guardar modelos
        import pickle
        with open(f"{models_dir}/qlearning_agent.pkl", 'wb') as f:
            pickle.dump(best_qlearning_agent, f)
        
        with open(f"{models_dir}/stochastic_agent.pkl", 'wb') as f:
            pickle.dump(best_stochastic_agent, f)
        
        # Preparar resultados para JSON
        results_summary = {
            'Express': {
                'qlearning': {
                    'best_params': qlearning_results['best_params'],
                    'best_score': qlearning_results['best_score'],
                    'evaluation': {k: [float(x) for x in v] for k, v in qlearning_eval.items()},
                    'training_rewards': [float(x) for x in qlearning_trainer.training_rewards]
                },
                'stochastic': {
                    'best_params': stochastic_results['best_params'],
                    'best_score': stochastic_results['best_score'],
                    'evaluation': {k: [float(x) for x in v] for k, v in stochastic_eval.items()},
                    'training_rewards': [float(x) for x in stochastic_trainer.training_rewards]
                }
            }
        }
        
        # Agregar metadatos
        results_summary['experiment_info'] = {
            'type': 'EXPRESS_EXPERIMENT',
            'objective': 'Validación rápida de mejoras automáticas',
            'config': EXPRESS_CONFIG,
            'environment': 'DescentEnv_REAL',
            'mejoras_aplicadas': [
                'RewardShaperAutomatico - supervivencia crítica',
                'Hiperparámetros ultra agresivos automáticos',
                'Discretización optimizada express',
                'Grid search focalizados en mejores parámetros',
                'Entrenamiento con mejoras automáticas',
            ]
        }
        
        # Guardar JSON
        with open('flan_results_express.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # === REPORTE FINAL ===
        print(f"\n✅ EXPERIMENTO EXPRESS COMPLETADO")
        print("=" * 80)
        print(f"⏱️ Tiempo total: {total_time / 3600:.1f} horas")
        print(f"📊 Resultados guardados en: flan_results_express.json")
        print(f"🎯 Modelos guardados en: {models_dir}/")
        
        # Mostrar resultados clave
        ql_reward = qlearning_eval['total_rewards']
        stoch_reward = stochastic_eval['total_rewards']
        
        print(f"\n📈 RESULTADOS CLAVE:")
        print(f"   Q-Learning promedio: {sum(ql_reward)/len(ql_reward):.2f}")
        print(f"   Stochastic promedio: {sum(stoch_reward)/len(stoch_reward):.2f}")
        print(f"   Supervivencia Q-Learning: {sum(qlearning_eval['survival_times'])/len(qlearning_eval['survival_times']):.1f} pasos")
        print(f"   Supervivencia Stochastic: {sum(stochastic_eval['survival_times'])/len(stochastic_eval['survival_times']):.1f} pasos")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ ERROR en experimento express: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal del experimento express"""
    print("🎯 FLAN - EXPERIMENTO EXPRESS")
    print("=" * 80)
    print("🔥 DescentEnv REAL + Mejoras Automáticas (Versión Optimizada)")
    print("📈 OBJETIVO: Validar mejoras automáticas rápidamente")
    print()
    
    # Información del sistema
    print("🖥️ INFORMACIÓN DEL SISTEMA")
    print("-" * 40)
    print(f"CPU cores: {psutil.cpu_count(logical=True)} lógicos")
    print(f"RAM disponible: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print()
    
    # Estimación de tiempo
    estimated_hours = estimate_express_time()
    
    # Confirmación
    print("🚨 CONFIGURACIÓN EXPRESS")
    print("=" * 50)
    print("• Entorno: DescentEnv REAL (BlueSky)")
    print("• Mejoras: Automáticas integradas")
    print("• Episodios: ~23,000 (vs 256,000 completo)")
    print(f"• Tiempo: ~{estimated_hours:.1f} horas (vs 120+ completo)")
    print("• Propósito: Validación científica rápida")
    print()
    
    confirm = input("¿Proceder con experimento EXPRESS? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ Experimento cancelado")
        return
    
    # Ejecutar
    success = run_express_experiment()
    
    if success:
        print("\n🎉 EXPERIMENTO EXPRESS COMPLETADO")
        print("📊 Revisa flan_results_express.json para resultados")
        print("\n🚀 SIGUIENTES PASOS:")
        print("   1. Analizar resultados express")
        print("   2. Si funciona bien, ejecutar experimento completo")
        print("   3. O ajustar configuración basándose en resultados")
    else:
        print("\n❌ El experimento express falló")

if __name__ == "__main__":
    main() 