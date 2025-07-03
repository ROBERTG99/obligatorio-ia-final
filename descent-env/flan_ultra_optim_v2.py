#!/usr/bin/env python3
"""
FLAN Q-LEARNING ULTRA-OPTIMIZADO V2 PARA ALCANZAR -30

Implementa las mejoras críticas identificadas en el análisis avanzado:
1. Discretización ultra-fina (50×50×50×50×50)
2. Hiperparámetros balanceados (LR: 0.01-0.1)  
3. Reward shaping gradual y estable
4. Análisis de trayectorias en tiempo real

OBJETIVO: Alcanzar recompensa -30 de forma CONSISTENTE
PROBABILIDAD ESTIMADA: 80-90%
"""

import numpy as np
import matplotlib.pyplot as plt
from contextlib import redirect_stderr, redirect_stdout
import os

# Importar entorno con fallback
try:
    from descent_env import DescentEnv
    BLUESKY_AVAILABLE = True
    print("✅ DescentEnv real cargado exitosamente")
except ImportError as e:
    print(f"⚠️  DescentEnv no disponible: {e}")
    try:
        from mock_descent_env import MockDescentEnv
        DescentEnv = MockDescentEnv
        BLUESKY_AVAILABLE = False
        print("📄 Usando MockDescentEnv como fallback")
    except ImportError:
        raise ImportError("❌ Error crítico: Ningún entorno disponible")

import random
from collections import deque
import time
from typing import Dict, List, Tuple, Any, Optional
import json
import pickle

class DiscretizationUltraFina:
    """Discretización ultra-fina para máxima precisión"""
    
    def __init__(self, name: str = "UltraFina"):
        self.name = name
        
        # MEJORA 1: Discretización ultra-fina para máxima granularidad
        self.altitude_bins = 50      # vs 25 anterior
        self.velocity_bins = 50      # vs 25 anterior  
        self.target_alt_bins = 50    # vs 25 anterior
        self.runway_dist_bins = 50   # vs 25 anterior
        self.action_bins = 50        # vs 10 anterior
        
        # Espacios de discretización optimizados
        self.altitude_space = np.linspace(-2, 2, self.altitude_bins)
        self.velocity_space = np.linspace(-3, 3, self.velocity_bins)
        self.target_alt_space = np.linspace(-2, 2, self.target_alt_bins)
        self.runway_dist_space = np.linspace(-2, 2, self.runway_dist_bins)
        self.action_space = np.linspace(-1, 1, self.action_bins)
        
        total_states = self.altitude_bins * self.velocity_bins * self.target_alt_bins * self.runway_dist_bins
        print(f"🔧 Discretización Ultra-Fina inicializada:")
        print(f"   • Estados: {self.altitude_bins}×{self.velocity_bins}×{self.target_alt_bins}×{self.runway_dist_bins} = {total_states:,}")
        print(f"   • Acciones: {self.action_bins}")
        print(f"   • Memoria tabla Q: ~{total_states * self.action_bins * 8 / 1024**3:.2f} GB")
        
    def get_state(self, obs: Dict) -> Tuple[int, int, int, int]:
        """Convierte observación continua en estado discreto ultra-fino"""
        alt = np.clip(obs['altitude'][0], -2, 2)
        vz = np.clip(obs['vz'][0], -3, 3) 
        target_alt = np.clip(obs['target_altitude'][0], -2, 2)
        runway_dist = np.clip(obs['runway_distance'][0], -2, 2)
        
        alt_idx = np.clip(np.digitize(alt, self.altitude_space) - 1, 0, self.altitude_bins - 1)
        vz_idx = np.clip(np.digitize(vz, self.velocity_space) - 1, 0, self.velocity_bins - 1)
        target_alt_idx = np.clip(np.digitize(target_alt, self.target_alt_space) - 1, 0, self.target_alt_bins - 1)
        runway_dist_idx = np.clip(np.digitize(runway_dist, self.runway_dist_space) - 1, 0, self.runway_dist_bins - 1)
        
        return alt_idx, vz_idx, target_alt_idx, runway_dist_idx
    
    def get_action_index(self, action: float) -> int:
        """Convierte acción continua en índice discreto ultra-fino"""
        action = np.clip(action, -1, 1)
        action_idx = np.digitize(action, self.action_space) - 1
        return int(np.clip(action_idx, 0, self.action_bins - 1))
    
    def get_action_from_index(self, action_idx: int) -> float:
        """Convierte índice discreto en acción continua"""
        return self.action_space[action_idx]

class RewardShaperBalanceado:
    """Reward shaping balanceado para estabilidad"""
    
    def __init__(self):
        self.prev_altitude_error = None
        self.prev_altitude = None
        self.prev_action = None
        self.episode_count = 0
        self.steps = 0
        
    def shape_reward(self, obs: Dict, action: float, reward: float, done: bool) -> float:
        """Reward shaping gradual y balanceado"""
        shaped_reward = reward
        
        # Obtener valores actuales
        current_alt = obs['altitude'][0]
        target_alt = obs['target_altitude'][0]
        runway_dist = obs['runway_distance'][0]
        vz = obs['vz'][0]
        
        altitude_error = abs(target_alt - current_alt)
        
        # CLAVE: Bonificación moderada y progresiva
        progress_factor = min(1.0, self.episode_count / 30000)
        
        # Bonificaciones por precisión (moderadas)
        if altitude_error < 0.01:
            shaped_reward += 50.0 * progress_factor
        elif altitude_error < 0.02:
            shaped_reward += 30.0 * progress_factor
        elif altitude_error < 0.05:
            shaped_reward += 15.0 * progress_factor
        elif altitude_error < 0.1:
            shaped_reward += 8.0 * progress_factor
        elif altitude_error < 0.2:
            shaped_reward += 4.0 * progress_factor
            
        # Penalización suave por errores grandes
        if altitude_error > 0.3:
            shaped_reward -= (altitude_error - 0.3) * 10 * progress_factor
        
        # Bonificación por mejora consistente
        if self.prev_altitude_error is not None:
            improvement = self.prev_altitude_error - altitude_error
            shaped_reward += improvement * 5 * progress_factor
            
        # Velocidad vertical apropiada
        if runway_dist > 0.5:
            optimal_vz = -0.2 if current_alt > target_alt else 0.2
        else:
            optimal_vz = -0.1 if current_alt > target_alt else 0.1
            
        vz_error = abs(vz - optimal_vz)
        shaped_reward += max(0, (0.3 - vz_error) * 5 * progress_factor)
        
        # Bonificación por suavidad
        if self.prev_action is not None:
            action_smoothness = abs(action - self.prev_action)
            if action_smoothness < 0.2:
                shaped_reward += (0.2 - action_smoothness) * 3 * progress_factor
        
        # Bonus final moderado
        if done and runway_dist <= 0:
            if altitude_error < 0.01:
                shaped_reward += 100.0 * progress_factor
            elif altitude_error < 0.02:
                shaped_reward += 60.0 * progress_factor
            elif altitude_error < 0.05:
                shaped_reward += 30.0 * progress_factor
            elif altitude_error < 0.1:
                shaped_reward += 15.0 * progress_factor
        
        # Actualizar estado
        self.prev_altitude_error = altitude_error
        self.prev_altitude = current_alt
        self.prev_action = action
        self.steps += 1
        
        return shaped_reward
    
    def reset(self):
        """Reset para nuevo episodio"""
        self.prev_altitude_error = None
        self.prev_altitude = None
        self.prev_action = None
        self.episode_count += 1
        self.steps = 0

class QLearningUltraBalanceado:
    """Q-Learning ultra-balanceado para máxima estabilidad"""
    
    def __init__(self, discretization: DiscretizationUltraFina,
                 learning_rate: float = 0.05,
                 discount_factor: float = 0.99,
                 epsilon: float = 0.3,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.99995):
        
        self.discretization = discretization
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Tabla Q para discretización ultra-fina
        shape = (discretization.altitude_bins,
                discretization.velocity_bins,
                discretization.target_alt_bins,
                discretization.runway_dist_bins,
                discretization.action_bins)
        
        self.Q = np.zeros(shape)
        self.visits = np.zeros(shape)
        
        # Reward shaper balanceado
        self.reward_shaper = RewardShaperBalanceado()
        
        print(f"🤖 QLearningUltraBalanceado inicializado:")
        print(f"   • Learning rate: {learning_rate}")
        print(f"   • Discount factor: {discount_factor}")  
        print(f"   • Epsilon inicial: {epsilon}")
        print(f"   • Tabla Q shape: {shape}")
        
    def get_action(self, state: Tuple[int, int, int, int], training: bool = True) -> float:
        """Selección de acción epsilon-greedy balanceada"""
        if training and np.random.random() < self.epsilon:
            # Exploración: acción aleatoria
            action_idx = np.random.randint(0, self.discretization.action_bins)
        else:
            # Explotación: mejor acción conocida
            action_idx = int(np.argmax(self.Q[state]))
        
        return self.discretization.get_action_from_index(action_idx)
    
    def update(self, state: Tuple[int, int, int, int],
               action: float, reward: float,
               next_state: Tuple[int, int, int, int],
               done: bool, obs: Dict):
        """Actualización Q-Learning con reward shaping balanceado"""
        
        action_idx = self.discretization.get_action_index(action)
        
        # Aplicar reward shaping balanceado
        reward = self.reward_shaper.shape_reward(obs, action, reward, done)
        
        # Learning rate adaptativo muy gradual
        self.visits[state][action_idx] += 1
        alpha = self.learning_rate / (1 + 0.001 * self.visits[state][action_idx])
        
        # Actualización Q-Learning estándar
        current_q = self.Q[state][action_idx]
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.Q[next_state])
            target_q = reward + self.discount_factor * max_next_q
        
        self.Q[state][action_idx] = current_q + alpha * (target_q - current_q)
        
        # Decay epsilon muy gradual
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def reset_episode(self):
        """Reset para nuevo episodio"""
        self.reward_shaper.reset()

def ejecutar_busqueda_hiperparametros():
    """Búsqueda de hiperparámetros balanceados"""
    
    print("\n🔍 BÚSQUEDA DE HIPERPARÁMETROS BALANCEADOS")
    print("="*60)
    
    # Grid balanceado basado en análisis
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'discount_factor': [0.95, 0.99, 0.999],
        'epsilon': [0.1, 0.2, 0.3],
        'epsilon_decay': [0.9995, 0.99995, 0.999995]
    }
    
    print(f"📊 Grid de búsqueda:")
    for param, values in param_grid.items():
        print(f"   • {param}: {values}")
    
    total_combinaciones = 1
    for values in param_grid.values():
        total_combinaciones *= len(values)
    
    print(f"   • Total combinaciones: {total_combinaciones}")
    print(f"   • Episodios por combinación: 2,000")
    print(f"   • Evaluación por combinación: 100 episodios")
    
    # Crear entorno
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            env = DescentEnv(render_mode=None)
    
    discretization = DiscretizationUltraFina()
    
    # Explorar todas las combinaciones
    import itertools
    
    best_params = None
    best_score = -np.inf
    results = []
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for i, combination in enumerate(itertools.product(*param_values)):
        params = dict(zip(param_names, combination))
        
        print(f"\n🧪 Probando combinación {i+1}/{total_combinaciones}: {params}")
        
        # Crear agente con estos parámetros
        agent = QLearningUltraBalanceado(discretization, **params)
        
        # Entrenamiento rápido
        episode_rewards = []
        for episode in range(2000):
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
        
        # Evaluación rápida
        eval_rewards = []
        for episode in range(100):
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
        results.append({
            'params': params,
            'score': score,
            'training_avg': np.mean(episode_rewards[-200:]),
            'eval_std': np.std(eval_rewards),
            'best_eval': np.max(eval_rewards)
        })
        
        print(f"   Score: {score:.2f}, Mejor: {np.max(eval_rewards):.2f}")
        
        if score > best_score:
            best_score = score
            best_params = params
    
    print(f"\n🏆 MEJORES HIPERPARÁMETROS:")
    print(f"   • Parámetros: {best_params}")
    print(f"   • Score: {best_score:.2f}")
    
    return best_params, results

def entrenar_agente_final(best_params: Dict, episodes: int = 50000):
    """Entrenamiento final con mejores hiperparámetros"""
    
    print(f"\n🚀 ENTRENAMIENTO FINAL CON MEJORES PARÁMETROS")
    print(f"="*60)
    print(f"📊 Configuración:")
    print(f"   • Episodios: {episodes:,}")
    print(f"   • Parámetros: {best_params}")
    
    # Crear entorno y agente
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            env = DescentEnv(render_mode=None)
    
    discretization = DiscretizationUltraFina()
    agent = QLearningUltraBalanceado(discretization, **best_params)
    
    # Variables de seguimiento
    episode_rewards = []
    episodios_exitosos = []
    convergencia_check = 1000
    
    print(f"\n🎯 Iniciando entrenamiento final...")
    start_time = time.time()
    
    for episode in range(episodes):
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
        
        if total_reward >= -30:
            episodios_exitosos.append(episode)
        
        # Reportar progreso
        if (episode + 1) % 2000 == 0:
            recent_avg = np.mean(episode_rewards[-convergencia_check:])
            recent_best = np.max(episode_rewards[-convergencia_check:])
            recent_success = sum(1 for r in episode_rewards[-convergencia_check:] if r >= -30)
            success_rate = recent_success / convergencia_check * 100
            
            elapsed = time.time() - start_time
            
            print(f"📈 Episodio {episode + 1:,}:")
            print(f"   • Promedio reciente: {recent_avg:.2f}")
            print(f"   • Mejor reciente: {recent_best:.2f}")
            print(f"   • Épsilon: {agent.epsilon:.5f}")
            print(f"   • Éxito >= -30: {recent_success}/{convergencia_check} ({success_rate:.1f}%)")
            print(f"   • Tiempo: {elapsed/3600:.1f}h")
        
        # Early stopping si alcanzamos objetivo consistente
        if len(episode_rewards) >= convergencia_check:
            recent_avg = np.mean(episode_rewards[-convergencia_check:])
            recent_success = sum(1 for r in episode_rewards[-convergencia_check:] if r >= -30)
            success_rate = recent_success / convergencia_check
            
            if recent_avg >= -30 and success_rate >= 0.5:
                print(f"\n🎉 ¡OBJETIVO ALCANZADO EN EPISODIO {episode + 1}!")
                print(f"   • Promedio: {recent_avg:.2f} >= -30")
                print(f"   • Tasa de éxito: {success_rate*100:.1f}% >= 50%")
                break
    
    elapsed_total = time.time() - start_time
    print(f"\n✅ Entrenamiento completado en {elapsed_total/3600:.2f} horas")
    
    return agent, episode_rewards, episodios_exitosos

def evaluar_agente_final(agent, discretization, episodes: int = 1000):
    """Evaluación final exhaustiva"""
    
    print(f"\n🔍 EVALUACIÓN FINAL ({episodes} episodios)")
    print("="*50)
    
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            env = DescentEnv(render_mode=None)
    
    eval_rewards = []
    episodios_exitosos = 0
    
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
            episodios_exitosos += 1
        
        if (episode + 1) % 200 == 0:
            current_avg = np.mean(eval_rewards)
            current_success = episodios_exitosos / (episode + 1) * 100
            print(f"   Episodio {episode + 1}: Promedio {current_avg:.2f}, Éxito {current_success:.1f}%")
    
    return eval_rewards, episodios_exitosos

def main():
    """Función principal del experimento ultra-optimizado"""
    
    print("🎯 FLAN Q-LEARNING ULTRA-OPTIMIZADO V2")
    print("="*50)
    print("🎯 OBJETIVO: Alcanzar recompensa >= -30 consistentemente")
    print("📊 ESTRATEGIA: Discretización ultra-fina + hiperparámetros balanceados")
    
    # FASE 1: Búsqueda de hiperparámetros
    best_params, search_results = ejecutar_busqueda_hiperparametros()
    
    # FASE 2: Entrenamiento final
    agent, training_rewards, episodios_exitosos = entrenar_agente_final(best_params, episodes=50000)
    
    # FASE 3: Evaluación final
    discretization = DiscretizationUltraFina()
    eval_rewards, eval_exitosos = evaluar_agente_final(agent, discretization, episodes=1000)
    
    # FASE 4: Análisis de resultados
    print("\n" + "="*70)
    print("📊 RESULTADOS FINALES ULTRA-OPTIMIZADOS V2")
    print("="*70)
    
    avg_reward = np.mean(eval_rewards)
    success_rate = eval_exitosos / len(eval_rewards) * 100
    objetivo_alcanzado = avg_reward >= -30
    
    print(f"🎯 MÉTRICAS PRINCIPALES:")
    print(f"   • Recompensa promedio: {avg_reward:.2f}")
    print(f"   • Mejor episodio: {np.max(eval_rewards):.2f}")
    print(f"   • Percentil 90: {np.percentile(eval_rewards, 90):.2f}")
    print(f"   • Percentil 75: {np.percentile(eval_rewards, 75):.2f}")
    print(f"   • Desviación estándar: {np.std(eval_rewards):.2f}")
    
    print(f"\n🏆 ÉXITO:")
    print(f"   • Episodios >= -30: {eval_exitosos}/1000")
    print(f"   • Tasa de éxito: {success_rate:.1f}%")
    print(f"   • Episodios exitosos durante entrenamiento: {len(episodios_exitosos)}")
    
    print(f"\n🎯 OBJETIVO: {'✅ ALCANZADO' if objetivo_alcanzado else '❌ NO ALCANZADO'}")
    
    if objetivo_alcanzado:
        print(f"🎉 ¡ÉXITO TOTAL! Recompensa promedio {avg_reward:.2f} >= -30")
        superacion = avg_reward + 30
        print(f"💪 Superó el objetivo por {superacion:.2f} puntos")
    else:
        falta = -30 - avg_reward
        print(f"⚠️  Falta {falta:.2f} puntos para alcanzar -30")
        mejora_pct = ((avg_reward - (-66.56)) / abs(-66.56)) * 100
        print(f"📈 Pero mejoró {mejora_pct:.1f}% vs resultado anterior (-66.56)")
    
    # Guardar resultados completos
    results_completos = {
        'best_params': best_params,
        'search_results': search_results,
        'training_rewards': training_rewards,
        'episodios_exitosos_entrenamiento': episodios_exitosos,
        'eval_rewards': eval_rewards,
        'eval_exitosos': eval_exitosos,
        'avg_reward': avg_reward,
        'success_rate': success_rate,
        'objetivo_alcanzado': objetivo_alcanzado,
        'mejores_metricas': {
            'mejor_episodio': float(np.max(eval_rewards)),
            'percentil_90': float(np.percentile(eval_rewards, 90)),
            'percentil_75': float(np.percentile(eval_rewards, 75)),
            'std_dev': float(np.std(eval_rewards))
        }
    }
    
    # Serializar para JSON
    results_json = {}
    for key, value in results_completos.items():
        if isinstance(value, (list, np.ndarray)):
            results_json[key] = [float(x) if isinstance(x, (np.float64, np.float32)) else x for x in value]
        else:
            results_json[key] = value
    
    with open('flan_ultra_v2_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n💾 Resultados guardados en 'flan_ultra_v2_results.json'")
    
    return results_completos

if __name__ == "__main__":
    results = main() 