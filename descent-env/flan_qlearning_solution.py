import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# OPTIMIZACIÓN: Usar DescentEnv real para máximo rendimiento
print("🚀 CONFIGURACIÓN OPTIMIZADA: Priorizando DescentEnv real para máximo rendimiento")

# Intentar importar DescentEnv real primero (verificado funcionando)
try:
    from descent_env import DescentEnv
    BLUESKY_AVAILABLE = True
    print("✅ DescentEnv real cargado exitosamente - MÁXIMO RENDIMIENTO")
except ImportError as e:
    print(f"⚠️  DescentEnv no disponible: {e}")
    BLUESKY_AVAILABLE = False

# MockDescentEnv solo como fallback extremo
try:
    from mock_descent_env import MockDescentEnv
    MOCK_AVAILABLE = True
except ImportError:
    print("Warning: MockDescentEnv tampoco disponible")
    MockDescentEnv = None
    MOCK_AVAILABLE = False

# Configuración optimizada: DescentEnv real siempre que sea posible
if BLUESKY_AVAILABLE:
    print("🎯 USANDO DESCENTENV REAL - Rendimiento óptimo garantizado")
    # DescentEnv ya importado
elif MOCK_AVAILABLE:
    print("📄 Fallback: Usando MockDescentEnv")
    DescentEnv = MockDescentEnv
else:
    raise ImportError("❌ Error crítico: Ningún entorno disponible")

import random
from collections import deque, namedtuple
import time
from typing import Dict, List, Tuple, Any, Optional, Union
import json
import pickle
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import signal
import sys
import heapq

# Variable global para manejo de señales
STOP_EXECUTION = False

def signal_handler(signum, frame):
    """Manejador de señales para parada graceful"""
    global STOP_EXECUTION
    print("\n\nRecibida señal de interrupción. Deteniendo ejecución...")
    STOP_EXECUTION = True
    sys.exit(0)

# Registrar manejador de señales
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Estructura para Prioritized Experience Replay
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'priority'])

class DiscretizationScheme:
    """Clase para manejar diferentes esquemas de discretización"""
    
    def __init__(self, name: str, 
                 altitude_bins: int, 
                 velocity_bins: int, 
                 target_alt_bins: int, 
                 runway_dist_bins: int,
                 action_bins: int):
        self.name = name
        self.altitude_bins = altitude_bins
        self.velocity_bins = velocity_bins
        self.target_alt_bins = target_alt_bins
        self.runway_dist_bins = runway_dist_bins
        self.action_bins = action_bins
        
        # Definir rangos de discretización basados en el análisis del entorno
        self.altitude_space = np.linspace(-2, 2, altitude_bins)  # Normalizado
        self.velocity_space = np.linspace(-3, 3, velocity_bins)  # Normalizado
        self.target_alt_space = np.linspace(-2, 2, target_alt_bins)  # Normalizado
        self.runway_dist_space = np.linspace(-2, 2, runway_dist_bins)  # Normalizado
        self.action_space = np.linspace(-1, 1, action_bins)
        
    def get_state(self, obs: Dict) -> Tuple[int, int, int, int]:
        """Convierte observación continua en estado discreto"""
        alt = obs['altitude'][0]
        vz = obs['vz'][0]
        target_alt = obs['target_altitude'][0]
        runway_dist = obs['runway_distance'][0]
        
        # Clamp valores fuera de rango
        alt = np.clip(alt, -2, 2)
        vz = np.clip(vz, -3, 3)
        target_alt = np.clip(target_alt, -2, 2)
        runway_dist = np.clip(runway_dist, -2, 2)
        
        alt_idx = np.digitize(alt, self.altitude_space) - 1
        vz_idx = np.digitize(vz, self.velocity_space) - 1
        target_alt_idx = np.digitize(target_alt, self.target_alt_space) - 1
        runway_dist_idx = np.digitize(runway_dist, self.runway_dist_space) - 1
        
        # Asegurar índices válidos
        alt_idx = np.clip(alt_idx, 0, self.altitude_bins - 1)
        vz_idx = np.clip(vz_idx, 0, self.velocity_bins - 1)
        target_alt_idx = np.clip(target_alt_idx, 0, self.target_alt_bins - 1)
        runway_dist_idx = np.clip(runway_dist_idx, 0, self.runway_dist_bins - 1)
        
        return alt_idx, vz_idx, target_alt_idx, runway_dist_idx
    
    def get_action_index(self, action: float) -> int:
        """Convierte acción continua en índice discreto"""
        action = np.clip(action, -1, 1)
        action_idx = np.digitize(action, self.action_space) - 1
        return int(np.clip(action_idx, 0, self.action_bins - 1))
    
    def get_action_from_index(self, action_idx: int) -> float:
        """Convierte índice discreto en acción continua"""
        return self.action_space[action_idx]

class QLearningAgent:
    """Agente Q-Learning mejorado con reward shaping y técnicas avanzadas"""
    
    def __init__(self, discretization: DiscretizationScheme, 
                 learning_rate: float = 0.1, 
                 discount_factor: float = 0.99, 
                 epsilon: float = 0.1,
                 use_double_q: bool = True,
                 use_reward_shaping: bool = True):
        self.discretization = discretization
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.use_double_q = use_double_q
        self.use_reward_shaping = use_reward_shaping
        
        # Inicializar tablas Q (doble para Double Q-Learning)
        shape = (discretization.altitude_bins, 
                discretization.velocity_bins, 
                discretization.target_alt_bins, 
                discretization.runway_dist_bins, 
                discretization.action_bins)
        self.Q1 = np.zeros(shape)
        self.Q2 = np.zeros(shape) if use_double_q else None
        
        # Contador de visitas para learning rate adaptativo
        self.visits = np.zeros(shape)
        
        # Reward shaper
        self.reward_shaper = RewardShaper() if use_reward_shaping else None
        
        # Epsilon decay
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def get_action(self, state: Tuple[int, int, int, int], training: bool = True) -> float:
        """Selecciona acción usando política epsilon-greedy mejorada"""
        if training and np.random.random() < self.epsilon:
            # Exploración: acción aleatoria
            action_idx = np.random.randint(0, self.discretization.action_bins)
        else:
            # Explotación: mejor acción según Q promedio (si Double Q-Learning)
            if self.use_double_q and self.Q2 is not None:
                q_values = (self.Q1[state] + self.Q2[state]) / 2
            else:
                q_values = self.Q1[state]
            action_idx = int(np.argmax(q_values))
        
        return self.discretization.get_action_from_index(action_idx)
    
    def update(self, state: Tuple[int, int, int, int], 
               action: float, reward: float, 
               next_state: Tuple[int, int, int, int], 
               done: bool, obs: Dict = None):
        """Actualiza la tabla Q usando técnicas mejoradas"""
        action_idx = self.discretization.get_action_index(action)
        
        # Aplicar reward shaping si está habilitado
        if self.reward_shaper is not None and obs is not None:
            reward = self.reward_shaper.shape_reward(obs, action, reward, done)
        
        # Learning rate adaptativo
        self.visits[state][action_idx] += 1
        alpha = self.learning_rate / (1 + 0.05 * self.visits[state][action_idx])
        
        # Double Q-Learning o Q-Learning estándar
        if self.use_double_q and self.Q2 is not None:
            # Alternar entre Q1 y Q2
            if np.random.random() < 0.5:
                # Actualizar Q1
                current_q = self.Q1[state][action_idx]
                if done:
                    target_q = reward
                else:
                    # Usar Q1 para seleccionar acción, Q2 para evaluar
                    best_action_idx = int(np.argmax(self.Q1[next_state]))
                    max_next_q = self.Q2[next_state][best_action_idx]
                    target_q = reward + self.discount_factor * max_next_q
                self.Q1[state][action_idx] = current_q + alpha * (target_q - current_q)
            else:
                # Actualizar Q2
                current_q = self.Q2[state][action_idx]
                if done:
                    target_q = reward
                else:
                    # Usar Q2 para seleccionar acción, Q1 para evaluar
                    best_action_idx = int(np.argmax(self.Q2[next_state]))
                    max_next_q = self.Q1[next_state][best_action_idx]
                    target_q = reward + self.discount_factor * max_next_q
                self.Q2[state][action_idx] = current_q + alpha * (target_q - current_q)
        else:
            # Q-Learning estándar mejorado
            current_q = self.Q1[state][action_idx]
            if done:
                target_q = reward
            else:
                max_next_q = np.max(self.Q1[next_state])
                target_q = reward + self.discount_factor * max_next_q
            self.Q1[state][action_idx] = current_q + alpha * (target_q - current_q)
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def reset_episode(self):
        """Reset para nuevo episodio"""
        if self.reward_shaper is not None:
            self.reward_shaper.reset()

class StochasticQLearningAgent:
    """Agente Stochastic Q-Learning mejorado para espacios de acción grandes"""
    
    def __init__(self, discretization: DiscretizationScheme,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.99,
                 epsilon: float = 0.1,
                 sample_size: int = 10,
                 use_reward_shaping: bool = True):
        self.discretization = discretization
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.sample_size = sample_size
        self.use_reward_shaping = use_reward_shaping
        
        # Inicializar tabla Q
        shape = (discretization.altitude_bins, 
                discretization.velocity_bins, 
                discretization.target_alt_bins, 
                discretization.runway_dist_bins, 
                discretization.action_bins)
        self.Q = np.zeros(shape)
        
        # Contador de visitas
        self.visits = np.zeros(shape)
        
        # Reward shaper
        self.reward_shaper = RewardShaper() if use_reward_shaping else None
        
        # Epsilon decay
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def get_action(self, state: Tuple[int, int, int, int], training: bool = True) -> float:
        """Selecciona acción usando Stochastic Q-Learning mejorado"""
        # Asegurar que sample_size no sea mayor que action_bins
        effective_sample_size = min(self.sample_size, self.discretization.action_bins)
        
        if training and np.random.random() < self.epsilon:
            # Exploración: muestrear acciones aleatorias
            sampled_indices = np.random.choice(
                self.discretization.action_bins, 
                size=effective_sample_size, 
                replace=False
            )
            # Seleccionar la mejor entre las muestreadas
            q_values = self.Q[state][sampled_indices]
            best_sampled_idx = sampled_indices[np.argmax(q_values)]
            action_idx = best_sampled_idx
        else:
            # Explotación: muestrear y seleccionar la mejor
            sampled_indices = np.random.choice(
                self.discretization.action_bins, 
                size=effective_sample_size, 
                replace=False
            )
            q_values = self.Q[state][sampled_indices]
            best_sampled_idx = sampled_indices[np.argmax(q_values)]
            action_idx = best_sampled_idx
        
        return self.discretization.get_action_from_index(action_idx)
    
    def update(self, state: Tuple[int, int, int, int], 
               action: float, reward: float, 
               next_state: Tuple[int, int, int, int], 
               done: bool, obs: Dict = None):
        """Actualiza la tabla Q usando Stochastic Q-Learning mejorado"""
        action_idx = self.discretization.get_action_index(action)
        
        # Aplicar reward shaping si está habilitado
        if self.reward_shaper is not None and obs is not None:
            reward = self.reward_shaper.shape_reward(obs, action, reward, done)
        
        # Learning rate adaptativo
        self.visits[state][action_idx] += 1
        alpha = self.learning_rate / (1 + 0.05 * self.visits[state][action_idx])
        
        # Stochastic Q-Learning update
        current_q = self.Q[state][action_idx]
        if done:
            target_q = reward
        else:
            # Muestrear acciones para el siguiente estado
            effective_sample_size = min(self.sample_size, self.discretization.action_bins)
            sampled_indices = np.random.choice(
                self.discretization.action_bins, 
                size=effective_sample_size, 
                replace=False
            )
            max_next_q = np.max(self.Q[next_state][sampled_indices])
            target_q = reward + self.discount_factor * max_next_q
        
        self.Q[state][action_idx] = current_q + alpha * (target_q - current_q)
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def reset_episode(self):
        """Reset para nuevo episodio"""
        if self.reward_shaper is not None:
            self.reward_shaper.reset()

class PerformanceEvaluator:
    """Evaluador de rendimiento del agente"""
    
    def __init__(self, env: DescentEnv, agent, discretization: DiscretizationScheme):
        self.env = env
        self.agent = agent
        self.discretization = discretization
        
    def evaluate_episode(self, render: bool = False) -> Dict[str, float]:
        """Evalúa un episodio completo"""
        obs, _ = self.env.reset()
        state = self.discretization.get_state(obs)
        
        total_reward = 0
        steps = 0
        max_altitude = -np.inf
        min_altitude = np.inf
        target_altitude = obs['target_altitude'][0]
        
        done = False
        while not done:
            action = self.agent.get_action(state, training=False)
            obs, reward, done, _, info = self.env.step(np.array([action]))
            next_state = self.discretization.get_state(obs)
            
            total_reward += reward
            steps += 1
            
            # Trackear métricas
            current_alt = obs['altitude'][0]
            max_altitude = max(max_altitude, current_alt)
            min_altitude = min(min_altitude, current_alt)
            
            state = next_state
            
            if render:
                self.env.render()
        
        # Calcular métricas adicionales
        altitude_error = abs(max_altitude - target_altitude)
        survival_time = steps
        
        return {
            'total_reward': total_reward,
            'steps': steps,
            'max_altitude': max_altitude,
            'min_altitude': min_altitude,
            'target_altitude': target_altitude,
            'altitude_error': altitude_error,
            'survival_time': survival_time,
            'final_altitude': info['final_altitude']
        }
    
    def evaluate_multiple_episodes(self, num_episodes: int = 100) -> Dict[str, List[float]]:
        """Evalúa múltiples episodios y retorna estadísticas"""
        results = {
            'total_rewards': [],
            'steps': [],
            'altitude_errors': [],
            'survival_times': [],
            'final_altitudes': []
        }
        
        for _ in range(num_episodes):
            episode_result = self.evaluate_episode()
            results['total_rewards'].append(episode_result['total_reward'])
            results['steps'].append(episode_result['steps'])
            results['altitude_errors'].append(episode_result['altitude_error'])
            results['survival_times'].append(episode_result['survival_time'])
            results['final_altitudes'].append(episode_result['final_altitude'])
        
        return results

def train_and_evaluate_agent(params_tuple):
    """Función para entrenar y evaluar un agente (para paralelización)"""
    agent_type, params, discretization_dict = params_tuple
    
    # Recrear discretización
    discretization = DiscretizationScheme(
        discretization_dict['name'],
        discretization_dict['altitude_bins'],
        discretization_dict['velocity_bins'],
        discretization_dict['target_alt_bins'],
        discretization_dict['runway_dist_bins'],
        discretization_dict['action_bins']
    )
    
    # OPTIMIZACIÓN: Usar DescentEnv real incluso en paralelización si está disponible
    if BLUESKY_AVAILABLE:
        print("🚀 Worker usando DescentEnv REAL para máximo rendimiento")
        env = DescentEnv(render_mode=None)
    elif MOCK_AVAILABLE:
        print("📄 Worker usando MockDescentEnv como fallback")
        env = MockDescentEnv(render_mode=None)
    else:
        raise ImportError("❌ No hay entornos disponibles para el worker")
    
    # Crear agente
    if agent_type == 'qlearning':
        agent = QLearningAgent(discretization, **params)
    else:
        agent = StochasticQLearningAgent(discretization, **params)
    
    # EPISODIOS COMPETITIVOS para búsqueda de hiperparámetros
    TRAINING_EPISODES = 800  # Aumentado de 200 a 800 (4x más)
    trainer = QLearningTrainer(env, agent, discretization)
    trainer.train(episodes=TRAINING_EPISODES, verbose=False)
    
    # Evaluación robusta para mejor estadística
    EVALUATION_EPISODES = 100  # Aumentado de 50 a 100 (2x más)
    evaluator = PerformanceEvaluator(env, agent, discretization)
    eval_results = evaluator.evaluate_multiple_episodes(num_episodes=EVALUATION_EPISODES)
    
    # Calcular score
    score = np.mean(eval_results['total_rewards'])
    
    return {
        'params': params,
        'score': score,
        'eval_results': eval_results,
        'training_episodes': TRAINING_EPISODES,
        'evaluation_episodes': EVALUATION_EPISODES
    }

class HyperparameterOptimizer:
    """Optimizador de hiperparámetros con paralelización"""
    
    def __init__(self, env: DescentEnv, discretization: DiscretizationScheme):
        self.env = env
        self.discretization = discretization
        
    def grid_search(self, agent_type: str = 'qlearning', 
                   param_grid: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """Realiza búsqueda en cuadrícula de hiperparámetros con paralelización"""
        
        if param_grid is None:
            if agent_type == 'qlearning':
                # Hiperparámetros optimizados para mejores recompensas
                param_grid = {
                    'learning_rate': [0.2, 0.3, 0.4, 0.5],  # Learning rates más altos
                    'discount_factor': [0.95, 0.98, 0.99],   # Más enfoque en recompensas futuras
                    'epsilon': [0.2, 0.3, 0.4],              # Más exploración inicial
                    'use_double_q': [True, False],            # Probar Double Q-Learning
                    'use_reward_shaping': [True]              # Siempre usar reward shaping
                }
            else:  # stochastic
                param_grid = {
                    'learning_rate': [0.2, 0.3, 0.4, 0.5],
                    'discount_factor': [0.95, 0.98, 0.99],
                    'epsilon': [0.2, 0.3, 0.4],
                    'sample_size': [5, 8, 10],
                    'use_reward_shaping': [True]
                }
        
        # Calcular número de combinaciones
        total_combinations = 1
        for param_values in param_grid.values():
            total_combinations *= len(param_values)
        
        print(f"Probando {total_combinations} combinaciones de hiperparámetros con paralelización...")
        
        # Obtener número de CPUs disponibles (USAR TODOS para máximo rendimiento)
        cpu_count = psutil.cpu_count(logical=True)
        if cpu_count is not None:
            num_cores = cpu_count  # USAR TODOS LOS CORES (cambiado de cpu_count - 1)
        else:
            num_cores = 2
        print(f"🚀 MÁXIMO RENDIMIENTO: Usando TODOS los {num_cores} cores de CPU disponibles")
        
        # Generar todas las combinaciones
        import itertools
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Preparar datos para paralelización
        discretization_dict = {
            'name': self.discretization.name,
            'altitude_bins': self.discretization.altitude_bins,
            'velocity_bins': self.discretization.velocity_bins,
            'target_alt_bins': self.discretization.target_alt_bins,
            'runway_dist_bins': self.discretization.runway_dist_bins,
            'action_bins': self.discretization.action_bins
        }
        
        tasks = []
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            tasks.append((agent_type, params, discretization_dict))
        
        # Ejecutar en paralelo
        results = []
        best_params = None
        best_score = -np.inf
        
        try:
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                # Enviar todas las tareas
                future_to_params = {executor.submit(train_and_evaluate_agent, task): task[1] 
                                   for task in tasks}
                
                # Procesar resultados conforme van completándose
                for i, future in enumerate(as_completed(future_to_params)):
                    if STOP_EXECUTION:
                        break
                        
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result['score'] > best_score:
                            best_score = result['score']
                            best_params = result['params']
                        
                        print(f"Completado {i+1}/{total_combinations}: {result['params']} -> Score: {result['score']:.2f}")
                        
                    except Exception as e:
                        params = future_to_params[future]
                        print(f"Error en combinación {params}: {e}")
                        
        except KeyboardInterrupt:
            print("Interrumpido por el usuario")
            
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }

class QLearningTrainer:
    """Entrenador mejorado para agentes Q-Learning"""
    
    def __init__(self, env, agent, discretization: DiscretizationScheme):
        self.env = env
        self.agent = agent
        self.discretization = discretization
        self.training_rewards = []
        
    def train(self, episodes: int = 1000, verbose: bool = True) -> List[float]:
        """Entrena el agente con técnicas mejoradas"""
        episode_rewards = []
        
        for episode in range(episodes):
            # Reset del agente para nuevo episodio
            if hasattr(self.agent, 'reset_episode'):
                self.agent.reset_episode()
                
            obs, _ = self.env.reset()
            state = self.discretization.get_state(obs)
            
            total_reward = 0
            done = False
            step = 0
            
            while not done and step < 500:  # Límite de pasos
                action = self.agent.get_action(state, training=True)
                next_obs, reward, done, _, _ = self.env.step(np.array([action]))
                next_state = self.discretization.get_state(next_obs)
                
                # Actualizar agente con observación para reward shaping
                if hasattr(self.agent, 'update'):
                    try:
                        # Intentar pasar observación para reward shaping
                        self.agent.update(state, action, reward, next_state, done, next_obs)
                    except TypeError:
                        # Fallback para agentes que no soportan obs
                        self.agent.update(state, action, reward, next_state, done)
                
                total_reward += reward
                state = next_state
                step += 1
            
            episode_rewards.append(total_reward)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                current_epsilon = getattr(self.agent, 'epsilon', 'N/A')
                print(f"Episodio {episode + 1}, Recompensa promedio (últimos 100): {avg_reward:.2f}, "
                      f"Epsilon: {current_epsilon:.3f}")
        
        self.training_rewards = episode_rewards
        return episode_rewards

class PrioritizedReplayBuffer:
    """Buffer de experiencia con priorización para mejorar el aprendizaje"""
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha  # Controla cuánto se usa la priorización
        self.buffer = []
        self.priorities = []
        self.pos = 0
        
    def add(self, state: Tuple, action: float, reward: float, 
            next_state: Tuple, done: bool, priority: float = None):
        """Agregar experiencia al buffer"""
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
            
        experience = Experience(state, action, reward, next_state, done, priority)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = experience
            self.priorities[self.pos] = priority
            
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """Muestrear experiencias basado en prioridades"""
        if len(self.buffer) < batch_size:
            return []
            
        # Calcular probabilidades basadas en prioridades
        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Muestrear índices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calcular importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        experiences = [self.buffer[i] for i in indices]
        return experiences, indices, weights
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Actualizar prioridades de experiencias"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)

class RewardShaper:
    """Clase para mejorar la función de recompensa con reward shaping"""
    
    def __init__(self):
        self.prev_altitude_error = None
        self.prev_altitude = None
        self.steps = 0
        
    def shape_reward(self, obs: Dict, action: float, reward: float, done: bool) -> float:
        """Aplicar reward shaping para mejorar la señal de aprendizaje"""
        shaped_reward = reward
        
        # Obtener valores actuales
        current_alt = obs['altitude'][0]
        target_alt = obs['target_altitude'][0]
        runway_dist = obs['runway_distance'][0]
        vz = obs['vz'][0]
        
        # 1. Recompensa por reducir error de altitud
        altitude_error = abs(target_alt - current_alt)
        if self.prev_altitude_error is not None:
            if altitude_error < self.prev_altitude_error:
                shaped_reward += 2.0  # Bonificación por mejorar
            else:
                shaped_reward -= 0.5  # Penalización menor por empeorar
        
        # 2. Recompensa por mantener velocidad vertical apropiada
        # Velocidad vertical óptima depende de la distancia a la pista
        if runway_dist > 0.5:  # Lejos de la pista
            optimal_vz = -0.2 if current_alt > target_alt else 0.2
        else:  # Cerca de la pista
            optimal_vz = -0.1 if current_alt > 0.1 else 0.0
            
        vz_error = abs(vz - optimal_vz)
        shaped_reward += 1.0 - vz_error * 2.0
        
        # 3. Bonificación por supervivencia
        if not done:
            shaped_reward += 0.5
            
        # 4. Penalización por acciones extremas
        shaped_reward -= abs(action) * 0.2
        
        # 5. Bonificación por estar cerca del target al final
        if done and runway_dist <= 0:
            if altitude_error < 0.1:
                shaped_reward += 20.0  # Gran bonificación por aterrizaje perfecto
            elif altitude_error < 0.2:
                shaped_reward += 10.0  # Bonificación moderada
            elif altitude_error < 0.3:
                shaped_reward += 5.0   # Bonificación pequeña
        
        # Actualizar estado previo
        self.prev_altitude_error = altitude_error
        self.prev_altitude = current_alt
        self.steps += 1
        
        return shaped_reward
    
    def reset(self):
        """Reset del reward shaper"""
        self.prev_altitude_error = None
        self.prev_altitude = None
        self.steps = 0

def main():
    """Función principal para ejecutar el experimento completo"""
    
    print("="*80)
    print("PROYECTO FLAN - CONFIGURACIÓN DE EPISODIOS COMPETITIVA")
    print("="*80)
    print("📊 BÚSQUEDA DE HIPERPARÁMETROS:")
    print("   • Entrenamiento por combinación: 800 episodios (4x incremento)")
    print("   • Evaluación por combinación: 100 episodios")
    print("\n🎯 ENTRENAMIENTO FINAL:")
    print("   • Entrenamiento del mejor agente: 5000 episodios (competitivo con 3k+ de otros)")
    print("   • Evaluación final: 500 episodios (estadísticamente robusto)")
    print("\n📈 BENCHMARKS DE REFERENCIA:")
    print("   • Profesor: 50,000 episodios")
    print("   • Otros compañeros: ~3,000 episodios")
    print("   • Nuestro objetivo: 5,000 episodios (superando la media)")
    print("\n🚀 MEJORAS IMPLEMENTADAS:")
    print("   • Reward Shaping: Activo")
    print("   • Double Q-Learning: Probando ambas variantes")
    print("   • Learning Rates optimizados: 0.2-0.5")
    print("   • Paralelización: 9 cores CPU")
    print("="*80)
    
    # CONFIGURACIÓN OPTIMIZADA: DescentEnv real prioritario para máximo rendimiento
    if BLUESKY_AVAILABLE:
        env = DescentEnv(render_mode=None)
        print("🚀 USANDO DESCENTENV REAL - Máximo rendimiento para entrenamiento final")
    elif MOCK_AVAILABLE:
        env = MockDescentEnv(render_mode=None)
        print("📄 Fallback: Usando MockDescentEnv para experimento")
    else:
        raise ImportError("❌ Error crítico: No hay entornos disponibles")
    
    # Definir esquemas de discretización
    discretization_schemes = [
        DiscretizationScheme("Fina", 50, 50, 50, 50, 20),
        DiscretizationScheme("Media", 25, 25, 25, 25, 10),
        DiscretizationScheme("Gruesa", 10, 10, 10, 10, 5)
    ]
    
    results_summary = {}
    
    for scheme in discretization_schemes:
        print(f"\n{'='*50}")
        print(f"Probando esquema de discretización: {scheme.name}")
        print(f"Bins: Alt={scheme.altitude_bins}, Vel={scheme.velocity_bins}, "
              f"Target={scheme.target_alt_bins}, Dist={scheme.runway_dist_bins}, "
              f"Actions={scheme.action_bins}")
        print(f"{'='*50}")
        
        # OPTIMIZACIÓN: Usar DescentEnv real incluso para hiperparámetros cuando sea posible
        print("\n1. Optimizando hiperparámetros para Q-Learning estándar...")
        print(f"   • Episodios por combinación: 800 (entrenamiento) + 100 (evaluación)")
        
        # Usar DescentEnv real prioritario para máximo rendimiento
        if BLUESKY_AVAILABLE:
            print("🚀 Optimizador usando DescentEnv REAL para máxima precisión")
            optimizer_env = DescentEnv(render_mode=None)
        elif MOCK_AVAILABLE:
            print("📄 Optimizador usando MockDescentEnv como fallback")
            optimizer_env = MockDescentEnv(render_mode=None)
        else:
            raise ImportError("❌ No hay entornos disponibles para optimización")
        
        optimizer = HyperparameterOptimizer(optimizer_env, scheme)
        qlearning_results = optimizer.grid_search('qlearning')
        
        # Entrenar mejor agente Q-Learning con entorno real
        FINAL_TRAINING_EPISODES = 5000  # COMPETITIVO: Aumentado de 1000 a 5000
        print(f"\n   • Entrenando mejor agente Q-Learning con {FINAL_TRAINING_EPISODES} episodios...")
        best_qlearning_agent = QLearningAgent(scheme, **qlearning_results['best_params'])
        qlearning_trainer = QLearningTrainer(env, best_qlearning_agent, scheme)
        qlearning_trainer.train(episodes=FINAL_TRAINING_EPISODES, verbose=True)
        
        # Evaluar Q-Learning con MÁS episodios
        FINAL_EVALUATION_EPISODES = 500  # ROBUSTO: Aumentado de 200 a 500
        print(f"\n   • Evaluando Q-Learning con {FINAL_EVALUATION_EPISODES} episodios...")
        qlearning_evaluator = PerformanceEvaluator(env, best_qlearning_agent, scheme)
        qlearning_eval = qlearning_evaluator.evaluate_multiple_episodes(num_episodes=FINAL_EVALUATION_EPISODES)
        
        # Optimizar hiperparámetros para Stochastic Q-Learning (usa MockDescentEnv en paralelo)
        print("\n2. Optimizando hiperparámetros para Stochastic Q-Learning...")
        print(f"   • Episodios por combinación: 800 (entrenamiento) + 100 (evaluación)")
        stochastic_results = optimizer.grid_search('stochastic')
        
        # Entrenar mejor agente Stochastic Q-Learning con entorno real
        print(f"\n   • Entrenando mejor agente Stochastic Q-Learning con {FINAL_TRAINING_EPISODES} episodios...")
        best_stochastic_agent = StochasticQLearningAgent(scheme, **stochastic_results['best_params'])
        stochastic_trainer = QLearningTrainer(env, best_stochastic_agent, scheme)
        stochastic_trainer.train(episodes=FINAL_TRAINING_EPISODES, verbose=True)
        
        # Evaluar Stochastic Q-Learning con MÁS episodios
        print(f"\n   • Evaluando Stochastic Q-Learning con {FINAL_EVALUATION_EPISODES} episodios...")
        stochastic_evaluator = PerformanceEvaluator(env, best_stochastic_agent, scheme)
        stochastic_eval = stochastic_evaluator.evaluate_multiple_episodes(num_episodes=FINAL_EVALUATION_EPISODES)
        
        # Guardar modelos entrenados
        models_dir = f"models_{scheme.name.lower()}"
        os.makedirs(models_dir, exist_ok=True)
        
        with open(f"{models_dir}/qlearning_agent.pkl", 'wb') as f:
            pickle.dump(best_qlearning_agent, f)
        
        with open(f"{models_dir}/stochastic_agent.pkl", 'wb') as f:
            pickle.dump(best_stochastic_agent, f)
        
        with open(f"{models_dir}/discretization.pkl", 'wb') as f:
            pickle.dump(scheme, f)
        
        print(f"Modelos guardados en {models_dir}/")
        
        # Guardar resultados
        results_summary[scheme.name] = {
            'qlearning': {
                'best_params': qlearning_results['best_params'],
                'best_score': qlearning_results['best_score'],
                'evaluation': qlearning_eval,
                'training_rewards': qlearning_trainer.training_rewards
            },
            'stochastic': {
                'best_params': stochastic_results['best_params'],
                'best_score': stochastic_results['best_score'],
                'evaluation': stochastic_eval,
                'training_rewards': stochastic_trainer.training_rewards
            }
        }
    
    # Guardar resultados
    with open('flan_results.json', 'w') as f:
        # Convertir numpy arrays a listas para serialización JSON
        serializable_results = {}
        for scheme_name, scheme_results in results_summary.items():
            serializable_results[scheme_name] = {}
            for agent_type, agent_results in scheme_results.items():
                serializable_results[scheme_name][agent_type] = {
                    'best_params': agent_results['best_params'],
                    'best_score': float(agent_results['best_score']),
                    'evaluation': {k: [float(x) for x in v] for k, v in agent_results['evaluation'].items()},
                    'training_rewards': [float(x) for x in agent_results['training_rewards']]
                }
        json.dump(serializable_results, f, indent=2)
    
    # Generar reporte
    generate_report(results_summary)
    
    return results_summary

def generate_report(results_summary: Dict):
    """Genera un reporte completo de los resultados"""
    
    print("\n" + "="*80)
    print("REPORTE FINAL - PROYECTO FLAN")
    print("="*80)
    
    # Comparar esquemas de discretización
    print("\n1. COMPARACIÓN DE ESQUEMAS DE DISCRETIZACIÓN")
    print("-" * 50)
    
    for scheme_name, scheme_results in results_summary.items():
        print(f"\nEsquema: {scheme_name}")
        
        ql_score = scheme_results['qlearning']['best_score']
        stoch_score = scheme_results['stochastic']['best_score']
        
        print(f"  Q-Learning estándar: {ql_score:.2f}")
        print(f"  Stochastic Q-Learning: {stoch_score:.2f}")
        print(f"  Mejora: {((stoch_score - ql_score) / abs(ql_score) * 100):.1f}%" if ql_score != 0 else "N/A")
        
        # Mejores hiperparámetros
        print(f"  Mejores hiperparámetros Q-Learning: {scheme_results['qlearning']['best_params']}")
        print(f"  Mejores hiperparámetros Stochastic: {scheme_results['stochastic']['best_params']}")
    
    # Análisis de rendimiento
    print("\n2. ANÁLISIS DE RENDIMIENTO")
    print("-" * 50)
    
    best_scheme = max(results_summary.keys(), 
                     key=lambda x: results_summary[x]['stochastic']['best_score'])
    
    print(f"Mejor esquema de discretización: {best_scheme}")
    
    best_results = results_summary[best_scheme]['stochastic']['evaluation']
    
    print(f"Recompensa promedio: {np.mean(best_results['total_rewards']):.2f} ± {np.std(best_results['total_rewards']):.2f}")
    print(f"Tiempo de supervivencia promedio: {np.mean(best_results['survival_times']):.1f} pasos")
    print(f"Error de altitud promedio: {np.mean(best_results['altitude_errors']):.3f}")
    
    # Generar gráficos
    generate_plots(results_summary)

def generate_plots(results_summary: Dict):
    """Genera gráficos de los resultados"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Comparación de scores por esquema
    schemes = list(results_summary.keys())
    ql_scores = [results_summary[s]['qlearning']['best_score'] for s in schemes]
    stoch_scores = [results_summary[s]['stochastic']['best_score'] for s in schemes]
    
    x = np.arange(len(schemes))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, ql_scores, width, label='Q-Learning Estándar', alpha=0.8)
    axes[0, 0].bar(x + width/2, stoch_scores, width, label='Stochastic Q-Learning', alpha=0.8)
    axes[0, 0].set_xlabel('Esquema de Discretización')
    axes[0, 0].set_ylabel('Score Promedio')
    axes[0, 0].set_title('Comparación de Rendimiento por Esquema')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(schemes)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Curvas de aprendizaje para el mejor esquema
    best_scheme = max(results_summary.keys(), 
                     key=lambda x: results_summary[x]['stochastic']['best_score'])
    
    ql_rewards = results_summary[best_scheme]['qlearning']['training_rewards']
    stoch_rewards = results_summary[best_scheme]['stochastic']['training_rewards']
    
    # Promedio móvil
    window = 50
    ql_smooth = np.convolve(ql_rewards, np.ones(window)/window, mode='valid')
    stoch_smooth = np.convolve(stoch_rewards, np.ones(window)/window, mode='valid')
    
    axes[0, 1].plot(ql_smooth, label='Q-Learning Estándar', alpha=0.8)
    axes[0, 1].plot(stoch_smooth, label='Stochastic Q-Learning', alpha=0.8)
    axes[0, 1].set_xlabel('Episodio')
    axes[0, 1].set_ylabel('Recompensa Promedio (ventana móvil)')
    axes[0, 1].set_title(f'Curvas de Aprendizaje - {best_scheme}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribución de recompensas finales
    best_results = results_summary[best_scheme]['stochastic']['evaluation']
    
    axes[1, 0].hist(best_results['total_rewards'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Recompensa Total')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].set_title(f'Distribución de Recompensas - {best_scheme}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Comparación de métricas
    metrics = ['total_rewards', 'survival_times', 'altitude_errors']
    metric_names = ['Recompensa Total', 'Tiempo de Supervivencia', 'Error de Altitud']
    
    ql_means = [np.mean(results_summary[best_scheme]['qlearning']['evaluation'][m]) for m in metrics]
    stoch_means = [np.mean(results_summary[best_scheme]['stochastic']['evaluation'][m]) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, ql_means, width, label='Q-Learning Estándar', alpha=0.8)
    axes[1, 1].bar(x + width/2, stoch_means, width, label='Stochastic Q-Learning', alpha=0.8)
    axes[1, 1].set_xlabel('Métrica')
    axes[1, 1].set_ylabel('Valor Promedio')
    axes[1, 1].set_title(f'Comparación de Métricas - {best_scheme}')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metric_names, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('flan_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    results = main()
