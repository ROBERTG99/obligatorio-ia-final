#!/usr/bin/env python3
"""
FLAN Q-Learning Optimizado para Alcanzar Recompensa -30
Implementa todas las mejoras identificadas en el análisis
"""

import numpy as np
import matplotlib.pyplot as plt
from contextlib import redirect_stderr, redirect_stdout
import os
import json

# Importar entorno optimizado
try:
    from descent_env import DescentEnv
    BLUESKY_AVAILABLE = True
    print("✅ DescentEnv real cargado - MÁXIMO RENDIMIENTO")
except ImportError as e:
    print(f"⚠️  DescentEnv no disponible: {e}")
    try:
        from mock_descent_env import MockDescentEnv
        DescentEnv = MockDescentEnv
        print("📄 Usando MockDescentEnv como fallback")
    except ImportError:
        raise ImportError("❌ Ningún entorno disponible")

import random
from collections import deque
import time
from typing import Dict, List, Tuple, Any, Optional
import pickle
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil

class DiscretizationSchemeOptimized:
    """Esquema de discretización optimizado para precisión"""
    
    def __init__(self, name: str = "Fina_Optimizada"):
        self.name = name
        # MEJORA 1: Discretización más fina para mayor precisión
        self.altitude_bins = 50
        self.velocity_bins = 50
        self.target_alt_bins = 50
        self.runway_dist_bins = 50
        self.action_bins = 20
        
        # Discretización adaptativa - más densa cerca de valores críticos
        self.altitude_space = self._create_adaptive_space(-2, 2, self.altitude_bins, focus_center=0)
        self.velocity_space = self._create_adaptive_space(-3, 3, self.velocity_bins, focus_center=0)
        self.target_alt_space = self._create_adaptive_space(-2, 2, self.target_alt_bins, focus_center=0)
        self.runway_dist_space = self._create_adaptive_space(-2, 2, self.runway_dist_bins, focus_center=0)
        self.action_space = np.linspace(-1, 1, self.action_bins)
        
    def _create_adaptive_space(self, min_val, max_val, bins, focus_center=0, focus_ratio=0.3):
        """Crea espacio de discretización adaptativo con mayor densidad cerca del centro"""
        # Crear espacio no uniforme con más bins cerca del focus_center
        total_range = max_val - min_val
        focus_range = total_range * focus_ratio
        
        # Mitad de los bins para la región de enfoque
        focus_bins = bins // 2
        outer_bins = bins - focus_bins
        
        # Región de enfoque (alta densidad)
        focus_min = max(min_val, focus_center - focus_range/2)
        focus_max = min(max_val, focus_center + focus_range/2)
        focus_space = np.linspace(focus_min, focus_max, focus_bins)
        
        # Regiones exteriores (baja densidad)
        left_space = np.linspace(min_val, focus_min, outer_bins//2) if focus_min > min_val else []
        right_space = np.linspace(focus_max, max_val, outer_bins//2) if focus_max < max_val else []
        
        # Combinar y ordenar
        all_space = np.concatenate([left_space, focus_space, right_space])
        return np.unique(np.sort(all_space))
    
    def get_state(self, obs: Dict) -> Tuple[int, int, int, int]:
        """Convierte observación continua en estado discreto"""
        alt = np.clip(obs['altitude'][0], -2, 2)
        vz = np.clip(obs['vz'][0], -3, 3)
        target_alt = np.clip(obs['target_altitude'][0], -2, 2)
        runway_dist = np.clip(obs['runway_distance'][0], -2, 2)
        
        alt_idx = np.digitize(alt, self.altitude_space) - 1
        vz_idx = np.digitize(vz, self.velocity_space) - 1
        target_alt_idx = np.digitize(target_alt, self.target_alt_space) - 1
        runway_dist_idx = np.digitize(runway_dist, self.runway_dist_space) - 1
        
        # Asegurar índices válidos
        alt_idx = np.clip(alt_idx, 0, len(self.altitude_space) - 1)
        vz_idx = np.clip(vz_idx, 0, len(self.velocity_space) - 1)
        target_alt_idx = np.clip(target_alt_idx, 0, len(self.target_alt_space) - 1)
        runway_dist_idx = np.clip(runway_dist_idx, 0, len(self.runway_dist_space) - 1)
        
        return alt_idx, vz_idx, target_alt_idx, runway_dist_idx
    
    def get_action_index(self, action: float) -> int:
        """Convierte acción continua en índice discreto"""
        action = np.clip(action, -1, 1)
        action_idx = np.digitize(action, self.action_space) - 1
        return int(np.clip(action_idx, 0, self.action_bins - 1))
    
    def get_action_from_index(self, action_idx: int) -> float:
        """Convierte índice discreto en acción continua"""
        return self.action_space[action_idx]

class RewardShaperOptimized:
    """Reward shaping optimizado para alcanzar -30"""
    
    def __init__(self):
        self.prev_altitude_error = None
        self.prev_altitude = None
        self.prev_action = None
        self.steps = 0
        self.trajectory_errors = deque(maxlen=10)
        
    def shape_reward(self, obs: Dict, action: float, reward: float, done: bool) -> float:
        """Reward shaping agresivo optimizado para -30"""
        shaped_reward = reward
        
        # Obtener valores actuales
        current_alt = obs['altitude'][0]
        target_alt = obs['target_altitude'][0]
        runway_dist = obs['runway_distance'][0]
        vz = obs['vz'][0]
        
        altitude_error = abs(target_alt - current_alt)
        self.trajectory_errors.append(altitude_error)
        
        # MEJORA 1: Bonificación exponencial agresiva por precisión de altitud
        if altitude_error < 0.02:  # Muy preciso
            shaped_reward += 100.0 * np.exp(-altitude_error * 50)
        elif altitude_error < 0.05:  # Preciso
            shaped_reward += 50.0 * np.exp(-altitude_error * 30)
        elif altitude_error < 0.1:  # Bueno
            shaped_reward += 25.0 * np.exp(-altitude_error * 20)
        elif altitude_error < 0.2:  # Aceptable
            shaped_reward += 10.0 * np.exp(-altitude_error * 10)
        
        # MEJORA 2: Penalización cuadrática agresiva por errores grandes
        if altitude_error > 0.3:
            shaped_reward -= (altitude_error - 0.3) ** 2 * 200
        elif altitude_error > 0.2:
            shaped_reward -= (altitude_error - 0.2) ** 2 * 100
            
        # MEJORA 3: Premio por consistencia en la trayectoria
        if len(self.trajectory_errors) >= 5:
            trajectory_std = np.std(list(self.trajectory_errors))
            if trajectory_std < 0.05:  # Trayectoria muy estable
                shaped_reward += 20.0
            elif trajectory_std < 0.1:  # Trayectoria estable
                shaped_reward += 10.0
                
        # MEJORA 4: Bonificación por suavidad de acciones
        if self.prev_action is not None:
            action_change = abs(action - self.prev_action)
            if action_change < 0.1:  # Acción suave
                shaped_reward += 15.0 * (0.1 - action_change) * 10
            elif action_change > 0.5:  # Acción brusca
                shaped_reward -= action_change * 20
                
        # MEJORA 5: Bonificación por progreso hacia el objetivo
        if self.prev_altitude_error is not None:
            error_improvement = self.prev_altitude_error - altitude_error
            if error_improvement > 0:  # Mejorando
                shaped_reward += error_improvement * 100
            else:  # Empeorando
                shaped_reward += error_improvement * 50
                
        # MEJORA 6: Bonificación por velocidad vertical apropiada
        if runway_dist > 0.5:  # Lejos de la pista
            optimal_vz = -0.3 if current_alt > target_alt else 0.3
        else:  # Cerca de la pista
            optimal_vz = -0.1 if current_alt > target_alt else 0.1
            
        vz_error = abs(vz - optimal_vz)
        shaped_reward += max(0, 10.0 - vz_error * 20)
        
        # MEJORA 7: Super bonificación por aterrizaje perfecto
        if done and runway_dist <= 0:
            if altitude_error < 0.02:
                shaped_reward += 500.0  # ¡MEGA BONIFICACIÓN!
            elif altitude_error < 0.05:
                shaped_reward += 200.0
            elif altitude_error < 0.1:
                shaped_reward += 100.0
            elif altitude_error < 0.2:
                shaped_reward += 50.0
                
        # Actualizar estado previo
        self.prev_altitude_error = altitude_error
        self.prev_altitude = current_alt
        self.prev_action = action
        self.steps += 1
        
        return shaped_reward
    
    def reset(self):
        """Reset del reward shaper"""
        self.prev_altitude_error = None
        self.prev_altitude = None
        self.prev_action = None
        self.steps = 0
        self.trajectory_errors.clear()

class EpsilonSchedulerOptimized:
    """Scheduler de epsilon optimizado con decay adaptativo"""
    
    def __init__(self, epsilon_start=0.95, epsilon_end=0.01, decay_episodes=2000, decay_type='exponential'):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_episodes = decay_episodes
        self.decay_type = decay_type
        
    def get_epsilon(self, episode):
        """Calcula epsilon para el episodio dado"""
        if episode >= self.decay_episodes:
            return self.epsilon_end
            
        progress = episode / self.decay_episodes
        
        if self.decay_type == 'exponential':
            # Decay exponencial con exploración inicial alta
            return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-progress * 6)
        elif self.decay_type == 'linear':
            return self.epsilon_start - progress * (self.epsilon_start - self.epsilon_end)
        else:  # polynomial
            return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1 - progress) ** 2

class QLearningAgentOptimized:
    """Q-Learning Agent optimizado específicamente para alcanzar -30"""
    
    def __init__(self, discretization: DiscretizationSchemeOptimized, 
                 learning_rate: float = 0.7, 
                 discount_factor: float = 0.999, 
                 epsilon_start: float = 0.95,
                 epsilon_end: float = 0.01,
                 decay_episodes: int = 2000):
        
        self.discretization = discretization
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # MEJORA: Epsilon scheduling avanzado
        self.epsilon_scheduler = EpsilonSchedulerOptimized(epsilon_start, epsilon_end, decay_episodes)
        
        # Inicializar tabla Q con optimistic initialization
        shape = (len(discretization.altitude_space), 
                len(discretization.velocity_space), 
                len(discretization.target_alt_space), 
                len(discretization.runway_dist_space), 
                discretization.action_bins)
        
        # MEJORA: Inicialización optimista para favorecer exploración
        self.Q = np.zeros(shape) + 0.1  # Pequeña inicialización optimista
        
        # Contadores y estadísticas
        self.visits = np.zeros(shape)
        self.episode_count = 0
        
        # Reward shaper optimizado
        self.reward_shaper = RewardShaperOptimized()
        
        # Learning rate adaptativo
        self.initial_lr = learning_rate
        
    def get_action(self, state: Tuple[int, int, int, int], training: bool = True) -> float:
        """Selección de acción con epsilon-greedy optimizado"""
        current_epsilon = self.epsilon_scheduler.get_epsilon(self.episode_count) if training else 0.0
        
        if training and np.random.random() < current_epsilon:
            # Exploración: acción aleatoria
            action_idx = np.random.randint(0, self.discretization.action_bins)
        else:
            # Explotación: mejor acción según Q-values
            action_idx = int(np.argmax(self.Q[state]))
        
        return self.discretization.get_action_from_index(action_idx)
    
    def update(self, state: Tuple[int, int, int, int], 
               action: float, reward: float, 
               next_state: Tuple[int, int, int, int], 
               done: bool, obs: Optional[Dict] = None):
        """Actualización Q-learning optimizada"""
        action_idx = self.discretization.get_action_index(action)
        
        # Aplicar reward shaping optimizado
        if obs is not None:
            reward = self.reward_shaper.shape_reward(obs, action, reward, done)
        
        # MEJORA: Learning rate adaptativo por visitas
        self.visits[state][action_idx] += 1
        visits = self.visits[state][action_idx]
        alpha = self.initial_lr / (1 + 0.01 * visits)  # Decay más lento
        
        # MEJORA: Q-learning con clip de gradientes
        current_q = self.Q[state][action_idx]
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.Q[next_state])
            target_q = reward + self.discount_factor * max_next_q
        
        # Clip del error para estabilidad
        td_error = target_q - current_q
        td_error = np.clip(td_error, -10, 10)  # Clip agresivo para estabilidad
        
        self.Q[state][action_idx] = current_q + alpha * td_error
    
    def reset_episode(self):
        """Reset para nuevo episodio"""
        self.episode_count += 1
        self.reward_shaper.reset()

def train_and_evaluate_optimized(params_tuple):
    """Función optimizada para entrenar y evaluar agente"""
    agent_params, discretization_dict = params_tuple
    
    # Recrear discretización
    discretization = DiscretizationSchemeOptimized()
    
    # Crear entorno con silenciado
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            env = DescentEnv(render_mode=None)
    
    # Crear agente optimizado
    agent = QLearningAgentOptimized(discretization, **agent_params)
    
    # ENTRENAMIENTO INTENSIVO OPTIMIZADO
    TRAINING_EPISODES = 3000  # Más episodios para convergencia
    
    episode_rewards = []
    for episode in range(TRAINING_EPISODES):
        agent.reset_episode()
        obs, _ = env.reset()
        state = discretization.get_state(obs)
        
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < 500:
            action = agent.get_action(state, training=True)
            next_obs, reward, done, _, _ = env.step(np.array([action]))
            next_state = discretization.get_state(next_obs)
            
            agent.update(state, action, reward, next_state, bool(done), next_obs)
            
            total_reward += reward
            state = next_state
            step += 1
        
        episode_rewards.append(total_reward)
    
    # EVALUACIÓN INTENSIVA
    EVALUATION_EPISODES = 200
    eval_rewards = []
    
    for _ in range(EVALUATION_EPISODES):
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
    
    return {
        'params': agent_params,
        'score': score,
        'eval_rewards': eval_rewards,
        'training_rewards': episode_rewards,
        'best_eval': np.max(eval_rewards),
        'worst_eval': np.min(eval_rewards),
        'success_rate': np.sum(np.array(eval_rewards) >= -30) / len(eval_rewards)
    }

def main_optimized():
    """Experimento principal optimizado para alcanzar -30"""
    
    print("="*80)
    print("🎯 EXPERIMENTO OPTIMIZADO PARA ALCANZAR RECOMPENSA -30")
    print("="*80)
    print("🚀 MEJORAS IMPLEMENTADAS:")
    print("   • Discretización Fina Adaptativa (50x50x50x50x20)")
    print("   • Reward Shaping Agresivo con bonificaciones exponenciales")
    print("   • Epsilon Decay Optimizado (0.95 → 0.01)")
    print("   • Learning Rate Adaptativo Mejorado")
    print("   • Entrenamiento Intensivo (3,000 episodios)")
    print("   • Evaluación Robusta (200 episodios)")
    print("="*80)
    
    # GRID SEARCH OPTIMIZADO PARA -30
    param_grid = [
        # Configuraciones agresivas para alcanzar -30
        {'learning_rate': 0.8, 'discount_factor': 0.999, 'epsilon_start': 0.95, 'epsilon_end': 0.01, 'decay_episodes': 1500},
        {'learning_rate': 0.7, 'discount_factor': 0.999, 'epsilon_start': 0.9, 'epsilon_end': 0.01, 'decay_episodes': 2000},
        {'learning_rate': 0.6, 'discount_factor': 0.998, 'epsilon_start': 0.95, 'epsilon_end': 0.005, 'decay_episodes': 1800},
        {'learning_rate': 0.9, 'discount_factor': 0.999, 'epsilon_start': 0.98, 'epsilon_end': 0.01, 'decay_episodes': 1200},
    ]
    
    print(f"🔍 PROBANDO {len(param_grid)} CONFIGURACIONES OPTIMIZADAS...")
    
    # Paralelización optimizada
    cpu_count = psutil.cpu_count(logical=True) or os.cpu_count()
    # Utilizar todos los cores lógicos disponibles para la búsqueda en paralelo
    num_cores = cpu_count if cpu_count and cpu_count > 0 else 2
    
    tasks = [(params, {}) for params in param_grid]
    
    results = []
    best_score = -np.inf
    best_config = None
    target_reached = False
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        future_to_params = {executor.submit(train_and_evaluate_optimized, task): task[0] 
                           for task in tasks}
        
        for i, future in enumerate(as_completed(future_to_params)):
            try:
                result = future.result()
                results.append(result)
                
                score = result['score']
                success_rate = result['success_rate']
                best_eval = result['best_eval']
                
                print(f"\n✅ Configuración {i+1}/{len(param_grid)} completada:")
                print(f"   • Parámetros: {result['params']}")
                print(f"   • Score promedio: {score:.2f}")
                print(f"   • Mejor episodio: {best_eval:.2f}")
                print(f"   • Tasa de éxito (-30): {success_rate*100:.1f}%")
                
                if best_eval >= -30:
                    print(f"   🎉 ¡OBJETIVO -30 ALCANZADO! Mejor: {best_eval:.2f}")
                    target_reached = True
                
                if score > best_score:
                    best_score = score
                    best_config = result
                    
            except Exception as e:
                print(f"❌ Error en configuración: {e}")
    
    # Resultados finales
    print(f"\n" + "="*80)
    print("📊 RESULTADOS FINALES")
    print("="*80)
    
    if best_config is None:
        print("⚠️  No se encontró una configuración válida durante la búsqueda.")
        return None

    print(f"🏆 MEJOR CONFIGURACIÓN:")
    print(f"   • Parámetros: {best_config['params']}")
    print(f"   • Score promedio: {best_config['score']:.2f}")
    print(f"   • Mejor episodio: {best_config['best_eval']:.2f}")
    print(f"   • Peor episodio: {best_config['worst_eval']:.2f}")
    print(f"   • Tasa de éxito (-30): {best_config['success_rate']*100:.1f}%")
    
    if target_reached:
        print(f"\n🎉 ¡ÉXITO! Objetivo -30 alcanzado")
    else:
        print(f"\n⚠️  Objetivo -30 no alcanzado consistentemente")
        print(f"   Mejor resultado: {max(r['best_eval'] for r in results):.2f}")
    
    # Guardar resultados
    results_optimized = {
        'best_config': best_config,
        'all_results': results,
        'target_reached': target_reached,
        'experiment_info': {
            'objective': -30,
            'configurations_tested': len(param_grid),
            'best_score': best_score,
            'optimization_features': [
                'Discretización Fina Adaptativa',
                'Reward Shaping Agresivo',
                'Epsilon Decay Optimizado',
                'Learning Rate Adaptativo',
                'Entrenamiento Intensivo'
            ]
        }
    }
    
    with open('results_optimized_target_30.json', 'w') as f:
        # Serializar numpy arrays
        for result in results_optimized['all_results']:
            result['eval_rewards'] = [float(x) for x in result['eval_rewards']]
            result['training_rewards'] = [float(x) for x in result['training_rewards']]
        
        json.dump(results_optimized, f, indent=2)
    
    print(f"\n💾 Resultados guardados en: results_optimized_target_30.json")
    
    return results_optimized

if __name__ == "__main__":
    results = main_optimized() 