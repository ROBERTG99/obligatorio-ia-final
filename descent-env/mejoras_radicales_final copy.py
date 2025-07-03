#!/usr/bin/env python3
"""
MEJORAS RADICALES FINALES PARA ALCANZAR -30

Basado en análisis de resultados actuales:
- Q-Learning actual: -66.56 (mejor episodio: -32.85)
- Objetivo: -30 (solo falta 2.85 puntos en el mejor caso!)
- Problema: INCONSISTENCIA, no capacidad

MEJORAS IMPLEMENTADAS:
1. Discretización ultra-fina (50×50×50×50×50)
2. Hiperparámetros balanceados (no extremos)
3. Reward shaping progresivo
4. Early stopping inteligente

ESTRATEGIA: Optimizar para CONSISTENCIA, no picos
"""

import numpy as np
import matplotlib.pyplot as plt
from contextlib import redirect_stderr, redirect_stdout
import os
import time
from typing import Dict, List, Tuple, Any, Optional
import json
import pickle

# Importar entorno
try:
    from descent_env import DescentEnv
    BLUESKY_AVAILABLE = True
    print("✅ DescentEnv real disponible")
except ImportError:
    try:
        from mock_descent_env import MockDescentEnv
        DescentEnv = MockDescentEnv
        BLUESKY_AVAILABLE = False
        print("📄 Usando MockDescentEnv")
    except ImportError:
        raise ImportError("❌ Ningún entorno disponible")

class DiscretizacionUltraFina:
    """Discretización 10x más fina para máxima precisión"""
    
    def __init__(self):
        # MEJORA 1: Discretización ultra-fina
        self.altitude_bins = 50      # vs 25 (2x más fino)
        self.velocity_bins = 50      # vs 25 (2x más fino)  
        self.target_alt_bins = 50    # vs 25 (2x más fino)
        self.runway_dist_bins = 50   # vs 25 (2x más fino)
        self.action_bins = 50        # vs 10 (5x más fino!)
        
        # Espacios optimizados para máxima cobertura
        self.altitude_space = np.linspace(-3, 3, self.altitude_bins)
        self.velocity_space = np.linspace(-4, 4, self.velocity_bins)
        self.target_alt_space = np.linspace(-3, 3, self.target_alt_bins)
        self.runway_dist_space = np.linspace(-3, 3, self.runway_dist_bins)
        self.action_space = np.linspace(-1, 1, self.action_bins)
        
        total_params = (self.altitude_bins * self.velocity_bins * 
                       self.target_alt_bins * self.runway_dist_bins * 
                       self.action_bins)
        
        print(f"🔧 DISCRETIZACIÓN ULTRA-FINA:")
        print(f"   • Estados: {self.altitude_bins}×{self.velocity_bins}×{self.target_alt_bins}×{self.runway_dist_bins}")
        print(f"   • Acciones: {self.action_bins}")
        print(f"   • Total parámetros: {total_params:,}")
        print(f"   • Mejora vs anterior: {total_params/(25*25*25*25*10):.1f}x más parámetros")
        
    def get_state(self, obs: Dict) -> Tuple[int, int, int, int]:
        """Discretización con máxima precisión"""
        alt = np.clip(obs['altitude'][0], -3, 3)
        vz = np.clip(obs['vz'][0], -4, 4) 
        target_alt = np.clip(obs['target_altitude'][0], -3, 3)
        runway_dist = np.clip(obs['runway_distance'][0], -3, 3)
        
        alt_idx = np.clip(np.digitize(alt, self.altitude_space) - 1, 0, self.altitude_bins - 1)
        vz_idx = np.clip(np.digitize(vz, self.velocity_space) - 1, 0, self.velocity_bins - 1)
        target_alt_idx = np.clip(np.digitize(target_alt, self.target_alt_space) - 1, 0, self.target_alt_bins - 1)
        runway_dist_idx = np.clip(np.digitize(runway_dist, self.runway_dist_space) - 1, 0, self.runway_dist_bins - 1)
        
        return alt_idx, vz_idx, target_alt_idx, runway_dist_idx
    
    def get_action_index(self, action: float) -> int:
        """Discretización de acción ultra-fina"""
        action = np.clip(action, -1, 1)
        action_idx = np.digitize(action, self.action_space) - 1
        return int(np.clip(action_idx, 0, self.action_bins - 1))
    
    def get_action_from_index(self, action_idx: int) -> float:
        """Convertir índice a acción"""
        return self.action_space[action_idx]

class RewardShaperProgresivo:
    """Reward shaping que progresa suavemente"""
    
    def __init__(self):
        self.episodio = 0
        self.prev_altitude_error = None
        self.prev_action = None
        
    def shape_reward(self, obs: Dict, action: float, reward: float, done: bool) -> float:
        """Reward shaping progresivo no extremo"""
        shaped_reward = reward
        
        # Métricas básicas
        current_alt = obs['altitude'][0]
        target_alt = obs['target_altitude'][0]
        runway_dist = obs['runway_distance'][0]
        vz = obs['vz'][0]
        altitude_error = abs(target_alt - current_alt)
        
        # Factor de progreso suave (no abrupto)
        progress = min(1.0, self.episodio / 25000)
        
        # Bonificaciones moderadas y progresivas
        if altitude_error < 0.01:
            shaped_reward += 25.0 * progress
        elif altitude_error < 0.02:
            shaped_reward += 15.0 * progress
        elif altitude_error < 0.05:
            shaped_reward += 10.0 * progress
        elif altitude_error < 0.1:
            shaped_reward += 5.0 * progress
        elif altitude_error < 0.2:
            shaped_reward += 2.0 * progress
        
        # Penalización suave por errores grandes
        if altitude_error > 0.5:
            shaped_reward -= (altitude_error - 0.5) * 5 * progress
        
        # Bonificación por mejora
        if self.prev_altitude_error is not None:
            improvement = self.prev_altitude_error - altitude_error
            shaped_reward += improvement * 3 * progress
        
        # Velocidad vertical apropiada
        if runway_dist > 0.5:
            target_vz = -0.15 if current_alt > target_alt else 0.15
        else:
            target_vz = -0.05 if current_alt > target_alt else 0.05
        
        vz_error = abs(vz - target_vz)
        shaped_reward += max(0, (0.3 - vz_error) * 3 * progress)
        
        # Suavidad de acciones
        if self.prev_action is not None:
            action_smoothness = abs(action - self.prev_action)
            if action_smoothness < 0.3:
                shaped_reward += (0.3 - action_smoothness) * 2 * progress
        
        # Bonus final moderado
        if done and runway_dist <= 0:
            if altitude_error < 0.01:
                shaped_reward += 75.0 * progress
            elif altitude_error < 0.02:
                shaped_reward += 50.0 * progress
            elif altitude_error < 0.05:
                shaped_reward += 30.0 * progress
            elif altitude_error < 0.1:
                shaped_reward += 15.0 * progress
        
        # Actualizar estado
        self.prev_altitude_error = altitude_error
        self.prev_action = action
        
        return shaped_reward
    
    def reset_episodio(self):
        """Reset episodio"""
        self.episodio += 1
        self.prev_altitude_error = None
        self.prev_action = None

class QLearningConsistente:
    """Q-Learning optimizado para consistencia"""
    
    def __init__(self, discretization,
                 learning_rate: float = 0.05,
                 discount_factor: float = 0.99,
                 epsilon: float = 0.25,
                 epsilon_min: float = 0.02,
                 epsilon_decay: float = 0.99995):
        
        self.discretization = discretization
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Tabla Q
        shape = (discretization.altitude_bins,
                discretization.velocity_bins,
                discretization.target_alt_bins,
                discretization.runway_dist_bins,
                discretization.action_bins)
        
        self.Q = np.zeros(shape)
        self.visits = np.zeros(shape)
        
        # Reward shaper
        self.reward_shaper = RewardShaperProgresivo()
        
        # Métricas
        self.episodios = 0
        self.recompensas_recientes = []
        
        print(f"🤖 QLearningConsistente:")
        print(f"   • Learning rate: {learning_rate}")
        print(f"   • Discount: {discount_factor}")
        print(f"   • Epsilon inicial: {epsilon}")
        
    def get_action(self, state: Tuple[int, int, int, int], training: bool = True) -> float:
        """Selección epsilon-greedy"""
        if training and np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, self.discretization.action_bins)
        else:
            action_idx = int(np.argmax(self.Q[state]))
        
        return self.discretization.get_action_from_index(action_idx)
    
    def update(self, state, action, reward, next_state, done, obs):
        """Actualización Q-Learning con reward shaping progresivo"""
        
        action_idx = self.discretization.get_action_index(action)
        
        # Reward shaping progresivo
        reward = self.reward_shaper.shape_reward(obs, action, reward, done)
        
        # Learning rate adaptativo conservador
        self.visits[state][action_idx] += 1
        alpha = self.learning_rate / (1 + 0.0005 * self.visits[state][action_idx])
        
        # Actualización Q
        current_q = self.Q[state][action_idx]
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.Q[next_state])
            target_q = reward + self.discount_factor * max_next_q
        
        self.Q[state][action_idx] = current_q + alpha * (target_q - current_q)
        
        # Epsilon decay gradual
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def finish_episode(self, reward):
        """Finalizar episodio"""
        self.episodios += 1
        self.recompensas_recientes.append(reward)
        if len(self.recompensas_recientes) > 1000:
            self.recompensas_recientes.pop(0)
        self.reward_shaper.reset_episodio()

def buscar_hiperparametros_optimos():
    """Búsqueda de hiperparámetros optimizada"""
    
    print("\n🔍 BÚSQUEDA DE HIPERPARÁMETROS OPTIMIZADA")
    print("="*50)
    
    # Grid balanceado (no extremo)
    param_grid = {
        'learning_rate': [0.03, 0.05, 0.08],      # Moderado
        'discount_factor': [0.98, 0.99],          # Estándar
        'epsilon': [0.2, 0.25, 0.3],              # Exploración adecuada
        'epsilon_decay': [0.9999, 0.99995]        # Muy gradual
    }
    
    print("📊 Grid balanceado:")
    for param, values in param_grid.items():
        print(f"   • {param}: {values}")
    
    import itertools
    combinaciones = list(itertools.product(*param_grid.values()))
    print(f"   • Combinaciones: {len(combinaciones)}")
    
    # Crear entorno
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            env = DescentEnv(render_mode=None)
    
    discretization = DiscretizacionUltraFina()
    
    best_params = None
    best_score = -np.inf
    results = []
    
    param_names = list(param_grid.keys())
    
    for i, combination in enumerate(combinaciones):
        params = dict(zip(param_names, combination))
        
        print(f"\n🧪 Probando {i+1}/{len(combinaciones)}: {params}")
        
        agent = QLearningConsistente(discretization, **params)
        
        # Entrenamiento rápido
        for episode in range(4000):
            obs, _ = env.reset()
            state = discretization.get_state(obs)
            
            total_reward = 0
            done = False
            step = 0
            
            while not done and step < 500:
                action = agent.get_action(state, training=True)
                next_obs, reward, done, _, _ = env.step(np.array([action]))
                next_state = discretization.get_state(next_obs)
                
                agent.update(state, action, reward, next_state, done, next_obs)
                
                total_reward += reward
                state = next_state
                step += 1
            
            agent.finish_episode(total_reward)
        
        # Evaluación
        eval_rewards = []
        for episode in range(300):
            obs, _ = env.reset()
            state = discretization.get_state(obs)
            
            total_reward = 0
            done = False
            step = 0
            
            while not done and step < 500:
                action = agent.get_action(state, training=False)
                obs, reward, done, _, _ = env.step(np.array([action]))
                next_state = discretization.get_state(obs)
                
                total_reward += reward
                state = next_state
                step += 1
            
            eval_rewards.append(total_reward)
        
        score = np.mean(eval_rewards)
        std_score = np.std(eval_rewards)
        best_eval = np.max(eval_rewards)
        
        results.append({
            'params': params,
            'score': score,
            'std': std_score,
            'best': best_eval,
            'consistency': -std_score  # Mejor consistencia = menor std
        })
        
        print(f"   📊 Score: {score:.2f}, Std: {std_score:.2f}, Mejor: {best_eval:.2f}")
        
        if score > best_score:
            best_score = score
            best_params = params
            print(f"   🏆 NUEVO MEJOR!")
    
    print(f"\n🏆 MEJORES HIPERPARÁMETROS:")
    print(f"   • Parámetros: {best_params}")
    print(f"   • Score: {best_score:.2f}")
    
    return best_params, results

def entrenar_modelo_final(params, max_episodes=60000):
    """Entrenamiento final con early stopping"""
    
    print(f"\n🚀 ENTRENAMIENTO FINAL")
    print("="*30)
    print(f"🎯 Objetivo: Alcanzar -30 consistentemente")
    print(f"📊 Parámetros: {params}")
    print(f"📈 Máximo episodios: {max_episodes:,}")
    
    # Crear entorno y agente
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            env = DescentEnv(render_mode=None)
    
    discretization = DiscretizacionUltraFina()
    agent = QLearningConsistente(discretization, **params)
    
    episode_rewards = []
    episodios_exitosos = []
    check_interval = 1000
    
    print(f"\n🎯 Iniciando entrenamiento...")
    start_time = time.time()
    
    for episode in range(max_episodes):
        obs, _ = env.reset()
        state = discretization.get_state(obs)
        
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < 500:
            action = agent.get_action(state, training=True)
            next_obs, reward, done, _, _ = env.step(np.array([action]))
            next_state = discretization.get_state(next_obs)
            
            agent.update(state, action, reward, next_state, done, next_obs)
            
            total_reward += reward
            state = next_state
            step += 1
        
        episode_rewards.append(total_reward)
        agent.finish_episode(total_reward)
        
        if total_reward >= -30:
            episodios_exitosos.append(episode)
        
        # Reporte cada 2500 episodios
        if (episode + 1) % 2500 == 0:
            recent_avg = np.mean(episode_rewards[-check_interval:])
            recent_best = np.max(episode_rewards[-check_interval:])
            recent_std = np.std(episode_rewards[-check_interval:])
            success_count = sum(1 for r in episode_rewards[-check_interval:] if r >= -30)
            success_rate = success_count / min(len(episode_rewards), check_interval) * 100
            
            elapsed = time.time() - start_time
            
            print(f"📈 Episodio {episode + 1:,}:")
            print(f"   • Promedio: {recent_avg:.2f}")
            print(f"   • Mejor: {recent_best:.2f}")
            print(f"   • Std: {recent_std:.2f}")
            print(f"   • Épsilon: {agent.epsilon:.5f}")
            print(f"   • Éxito >= -30: {success_count}/{check_interval} ({success_rate:.1f}%)")
            print(f"   • Tiempo: {elapsed/3600:.1f}h")
        
        # Early stopping
        if len(episode_rewards) >= check_interval:
            recent_avg = np.mean(episode_rewards[-check_interval:])
            success_count = sum(1 for r in episode_rewards[-check_interval:] if r >= -30)
            success_rate = success_count / check_interval
            
            # Condición de éxito: promedio >= -30 Y al menos 25% de éxito
            if recent_avg >= -30 and success_rate >= 0.25:
                print(f"\n🎉 ¡OBJETIVO ALCANZADO EN EPISODIO {episode + 1}!")
                print(f"   • Promedio: {recent_avg:.2f} >= -30")
                print(f"   • Tasa éxito: {success_rate*100:.1f}% >= 25%")
                break
            
            # Early stopping por convergencia después de 40k episodios
            elif episode >= 40000:
                recent_std = np.std(episode_rewards[-2000:])
                if recent_std < 8.0:
                    print(f"\n⚠️  Early stopping por convergencia")
                    print(f"   • Promedio: {recent_avg:.2f}")
                    print(f"   • Std: {recent_std:.2f} < 8.0")
                    break
    
    elapsed_total = time.time() - start_time
    print(f"\n✅ Entrenamiento completado en {elapsed_total/3600:.2f}h")
    
    return agent, episode_rewards, episodios_exitosos

def evaluar_modelo_final(agent, discretization, episodes=1500):
    """Evaluación final robusta"""
    
    print(f"\n🔍 EVALUACIÓN FINAL ({episodes} episodios)")
    print("="*40)
    
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            env = DescentEnv(render_mode=None)
    
    eval_rewards = []
    exitosos = 0
    
    for episode in range(episodes):
        obs, _ = env.reset()
        state = discretization.get_state(obs)
        
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < 500:
            action = agent.get_action(state, training=False)
            obs, reward, done, _, _ = env.step(np.array([action]))
            next_state = discretization.get_state(obs)
            
            total_reward += reward
            state = next_state
            step += 1
        
        eval_rewards.append(total_reward)
        
        if total_reward >= -30:
            exitosos += 1
        
        if (episode + 1) % 300 == 0:
            avg = np.mean(eval_rewards)
            rate = exitosos / (episode + 1) * 100
            print(f"   📊 {episode + 1}: Promedio {avg:.2f}, Éxito {rate:.1f}%")
    
    return eval_rewards, exitosos

def main():
    """Función principal de mejoras radicales finales"""
    
    print("🎯 MEJORAS RADICALES FINALES PARA ALCANZAR -30")
    print("="*55)
    print("🚨 SITUACIÓN: Q-Learning actual -66.56, mejor episodio -32.85")
    print("🎯 OBJETIVO: -30 (solo 2.85 puntos del mejor episodio)")
    print("💡 ESTRATEGIA: Optimizar CONSISTENCIA, no picos")
    
    # FASE 1: Búsqueda de hiperparámetros
    print(f"\n{'='*60}")
    print("FASE 1: BÚSQUEDA DE HIPERPARÁMETROS")
    print('='*60)
    
    best_params, search_results = buscar_hiperparametros_optimos()
    
    # FASE 2: Entrenamiento final
    print(f"\n{'='*60}")
    print("FASE 2: ENTRENAMIENTO FINAL")
    print('='*60)
    
    agent, training_rewards, episodios_exitosos = entrenar_modelo_final(best_params)
    
    # FASE 3: Evaluación final
    print(f"\n{'='*60}")
    print("FASE 3: EVALUACIÓN FINAL")
    print('='*60)
    
    discretization = DiscretizacionUltraFina()
    eval_rewards, eval_exitosos = evaluar_modelo_final(agent, discretization)
    
    # FASE 4: Análisis de resultados
    print(f"\n{'='*70}")
    print("📊 RESULTADOS FINALES - MEJORAS RADICALES")
    print('='*70)
    
    avg_reward = np.mean(eval_rewards)
    success_rate = eval_exitosos / len(eval_rewards) * 100
    objetivo_alcanzado = avg_reward >= -30
    
    # Comparación con situación anterior
    anterior = -66.56
    mejora = avg_reward - anterior
    mejora_pct = (mejora / abs(anterior)) * 100
    
    print(f"🎯 COMPARACIÓN CON SITUACIÓN ANTERIOR:")
    print(f"   • Resultado anterior: {anterior:.2f}")
    print(f"   • Resultado actual: {avg_reward:.2f}")
    print(f"   • Mejora absoluta: +{mejora:.2f} puntos")
    print(f"   • Mejora porcentual: +{mejora_pct:.1f}%")
    
    print(f"\n📊 MÉTRICAS DETALLADAS:")
    print(f"   • Mejor episodio: {np.max(eval_rewards):.2f}")
    print(f"   • Peor episodio: {np.min(eval_rewards):.2f}")
    print(f"   • Percentil 90: {np.percentile(eval_rewards, 90):.2f}")
    print(f"   • Percentil 75: {np.percentile(eval_rewards, 75):.2f}")
    print(f"   • Percentil 25: {np.percentile(eval_rewards, 25):.2f}")
    print(f"   • Desviación estándar: {np.std(eval_rewards):.2f}")
    
    print(f"\n🏆 ÉXITO:")
    print(f"   • Episodios >= -30: {eval_exitosos}/{len(eval_rewards)}")
    print(f"   • Tasa de éxito: {success_rate:.1f}%")
    print(f"   • Episodios exitosos en entrenamiento: {len(episodios_exitosos)}")
    
    print(f"\n🎯 OBJETIVO: {'✅ ALCANZADO' if objetivo_alcanzado else '❌ NO ALCANZADO'}")
    
    if objetivo_alcanzado:
        superacion = avg_reward + 30
        print(f"🎉 ¡ÉXITO TOTAL! Promedio {avg_reward:.2f} >= -30")
        print(f"🚀 Superó objetivo por {superacion:.2f} puntos")
        print(f"💪 Las mejoras radicales FUNCIONARON perfectamente")
    else:
        falta = -30 - avg_reward
        print(f"⚠️  Falta {falta:.2f} puntos para alcanzar -30")
        
        if mejora >= 25:
            print(f"🎉 MEJORA EXCELENTE: +{mejora:.2f} puntos")
            print(f"📈 Continuar con más entrenamiento o ensemble")
        elif mejora >= 15:
            print(f"👍 MEJORA BUENA: +{mejora:.2f} puntos")
            print(f"🔧 Ajustar reward shaping o probar más episodios")
        elif mejora >= 5:
            print(f"📊 MEJORA MODERADA: +{mejora:.2f} puntos")
            print(f"🤔 Considerar algoritmos deep learning")
        else:
            print(f"😞 MEJORA PEQUEÑA: +{mejora:.2f} puntos")
            print(f"🔄 Cambio de paradigma necesario (DDPG/TD3/SAC)")
    
    # Guardar resultados
    results = {
        'config': {
            'discretization': 'ultra_fina_50x50x50x50x50',
            'reward_shaping': 'progresivo',
            'hiperparametros': 'balanceados'
        },
        'search_results': search_results,
        'best_params': best_params,
        'training_rewards': training_rewards,
        'episodios_exitosos_training': episodios_exitosos,
        'eval_rewards': eval_rewards,
        'eval_exitosos': eval_exitosos,
        'metricas_finales': {
            'avg_reward': float(avg_reward),
            'success_rate': float(success_rate),
            'objetivo_alcanzado': objetivo_alcanzado,
            'mejora_absoluta': float(mejora),
            'mejora_porcentual': float(mejora_pct),
            'mejor_episodio': float(np.max(eval_rewards)),
            'percentil_90': float(np.percentile(eval_rewards, 90)),
            'std_dev': float(np.std(eval_rewards))
        }
    }
    
         # Serializar para JSON
    def serialize_for_json(obj):
         if isinstance(obj, list) or isinstance(obj, np.ndarray):
             return [float(x) if isinstance(x, (np.float64, np.float32, np.int64)) else x for x in obj]
         elif isinstance(obj, dict):
             return {k: serialize_for_json(v) for k, v in obj.items()}
         else:
             return obj
    
    results_json = serialize_for_json(results)
    
    with open('mejoras_radicales_final_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n💾 Resultados guardados en 'mejoras_radicales_final_results.json'")
    
    # Guardar modelo si es exitoso
    if objetivo_alcanzado or mejora >= 20:
        with open('modelo_final_exitoso.pkl', 'wb') as f:
            pickle.dump({
                'agent': agent,
                'discretization': discretization,
                'params': best_params,
                'performance': avg_reward
            }, f)
        print(f"💾 Modelo exitoso guardado en 'modelo_final_exitoso.pkl'")
    
    print(f"\n📋 RECOMENDACIONES FINALES:")
    if objetivo_alcanzado:
        print(f"   ✅ Objetivo alcanzado - Proyecto completado exitosamente")
        print(f"   📊 Generar reporte final y documentación")
        print(f"   🔬 Analizar factores de éxito para futuros proyectos")
    elif mejora >= 20:
        print(f"   🔄 Continuar entrenamiento por más episodios (100k+)")
        print(f"   🎯 Ajustar reward shaping para fase final")
        print(f"   📊 Probar ensemble de múltiples agentes")
    else:
        print(f"   🧠 Migrar a algoritmos deep learning (DDPG/TD3/SAC)")
        print(f"   🔄 Q-Learning tabular puede haber alcanzado límite teórico")
        print(f"   📚 Investigar control continuo y function approximation")
    
    return results

if __name__ == "__main__":
    results = main()
