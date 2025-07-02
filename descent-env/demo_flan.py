#!/usr/bin/env python3
"""
Demo simplificado del proyecto FLAN - Q-Learning para control de descenso de aeronave
"""

import numpy as np
import matplotlib.pyplot as plt

# Importación condicional del entorno
try:
    from descent_env import DescentEnv
    print("Usando DescentEnv con BlueSky")
except ImportError:
    from mock_descent_env import MockDescentEnv as DescentEnv
    print("BlueSky no disponible - usando MockDescentEnv")

import random
from typing import Dict, List, Tuple

class SimpleDiscretization:
    """Discretización simple para demostración"""
    
    def __init__(self, name: str, bins: int = 10):
        self.name = name
        self.bins = bins
        
        # Espacios de discretización simplificados
        self.altitude_space = np.linspace(-1, 1, bins)
        self.velocity_space = np.linspace(-1, 1, bins)
        self.target_alt_space = np.linspace(-1, 1, bins)
        self.runway_dist_space = np.linspace(-1, 1, bins)
        self.action_space = np.linspace(-1, 1, 5)  # 5 acciones discretas
        
    def get_state(self, obs: Dict) -> Tuple[int, int, int, int]:
        """Convierte observación en estado discreto"""
        alt = np.clip(obs['altitude'][0], -1, 1)
        vz = np.clip(obs['vz'][0], -1, 1)
        target_alt = np.clip(obs['target_altitude'][0], -1, 1)
        runway_dist = np.clip(obs['runway_distance'][0], -1, 1)
        
        alt_idx = np.digitize(alt, self.altitude_space) - 1
        vz_idx = np.digitize(vz, self.velocity_space) - 1
        target_alt_idx = np.digitize(target_alt, self.target_alt_space) - 1
        runway_dist_idx = np.digitize(runway_dist, self.runway_dist_space) - 1
        
        # Asegurar índices válidos
        alt_idx = np.clip(alt_idx, 0, self.bins - 1)
        vz_idx = np.clip(vz_idx, 0, self.bins - 1)
        target_alt_idx = np.clip(target_alt_idx, 0, self.bins - 1)
        runway_dist_idx = np.clip(runway_dist_idx, 0, self.bins - 1)
        
        return alt_idx, vz_idx, target_alt_idx, runway_dist_idx
    
    def get_action_index(self, action: float) -> int:
        """Convierte acción continua en índice discreto"""
        action = np.clip(action, -1, 1)
        action_idx = np.digitize(action, self.action_space) - 1
        return int(np.clip(action_idx, 0, len(self.action_space) - 1))
    
    def get_action_from_index(self, action_idx: int) -> float:
        """Convierte índice discreto en acción continua"""
        return self.action_space[action_idx]

class SimpleQLearningAgent:
    """Agente Q-Learning simplificado"""
    
    def __init__(self, discretization: SimpleDiscretization, 
                 learning_rate: float = 0.1, 
                 discount_factor: float = 0.99, 
                 epsilon: float = 0.1):
        self.discretization = discretization
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Tabla Q simplificada
        shape = (discretization.bins, discretization.bins, 
                discretization.bins, discretization.bins, len(discretization.action_space))
        self.Q = np.zeros(shape)
        
    def get_action(self, state: Tuple[int, int, int, int], training: bool = True) -> float:
        """Selecciona acción usando política epsilon-greedy"""
        if training and np.random.random() < self.epsilon:
            # Exploración
            action_idx = np.random.randint(0, len(self.discretization.action_space))
        else:
            # Explotación
            action_idx = int(np.argmax(self.Q[state]))
        
        return self.discretization.get_action_from_index(action_idx)
    
    def update(self, state: Tuple[int, int, int, int], 
               action: float, reward: float, 
               next_state: Tuple[int, int, int, int], 
               done: bool):
        """Actualiza la tabla Q"""
        action_idx = self.discretization.get_action_index(action)
        
        current_q = self.Q[state][action_idx]
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.Q[next_state])
            target_q = reward + self.discount_factor * max_next_q
        
        self.Q[state][action_idx] = current_q + self.learning_rate * (target_q - current_q)

class SimpleStochasticQLearningAgent:
    """Agente Stochastic Q-Learning simplificado"""
    
    def __init__(self, discretization: SimpleDiscretization,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.99,
                 epsilon: float = 0.1,
                 sample_size: int = 3):
        self.discretization = discretization
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.sample_size = min(sample_size, len(discretization.action_space))
        
        # Tabla Q simplificada
        shape = (discretization.bins, discretization.bins, 
                discretization.bins, discretization.bins, len(discretization.action_space))
        self.Q = np.zeros(shape)
        
    def get_action(self, state: Tuple[int, int, int, int], training: bool = True) -> float:
        """Selecciona acción usando Stochastic Q-Learning"""
        if training and np.random.random() < self.epsilon:
            # Exploración: muestrear acciones
            sampled_indices = np.random.choice(
                len(self.discretization.action_space), 
                size=self.sample_size, 
                replace=False
            )
            q_values = self.Q[state][sampled_indices]
            best_sampled_idx = sampled_indices[np.argmax(q_values)]
            action_idx = best_sampled_idx
        else:
            # Explotación: muestrear y seleccionar la mejor
            sampled_indices = np.random.choice(
                len(self.discretization.action_space), 
                size=self.sample_size, 
                replace=False
            )
            q_values = self.Q[state][sampled_indices]
            best_sampled_idx = sampled_indices[np.argmax(q_values)]
            action_idx = best_sampled_idx
        
        return self.discretization.get_action_from_index(action_idx)
    
    def update(self, state: Tuple[int, int, int, int], 
               action: float, reward: float, 
               next_state: Tuple[int, int, int, int], 
               done: bool):
        """Actualiza la tabla Q usando Stochastic Q-Learning"""
        action_idx = self.discretization.get_action_index(action)
        
        current_q = self.Q[state][action_idx]
        if done:
            target_q = reward
        else:
            # Muestrear acciones para el siguiente estado
            sampled_indices = np.random.choice(
                len(self.discretization.action_space), 
                size=self.sample_size, 
                replace=False
            )
            max_next_q = np.max(self.Q[next_state][sampled_indices])
            target_q = reward + self.discount_factor * max_next_q
        
        self.Q[state][action_idx] = current_q + self.learning_rate * (target_q - current_q)

def train_agent(env: DescentEnv, agent, discretization: SimpleDiscretization, 
                episodes: int = 200, verbose: bool = True) -> List[float]:
    """Entrena un agente"""
    episode_rewards = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        state = discretization.get_state(obs)
        
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state, training=True)
            obs, reward, done, _, _ = env.step(np.array([action]))
            next_state = discretization.get_state(obs)
            
            # Actualizar agente
            agent.update(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
        
        episode_rewards.append(total_reward)
        
        if verbose and (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episodio {episode + 1}, Recompensa promedio (últimos 50): {avg_reward:.2f}")
    
    return episode_rewards

def evaluate_agent(env: DescentEnv, agent, discretization: SimpleDiscretization, 
                  num_episodes: int = 50) -> Dict[str, List[float]]:
    """Evalúa un agente entrenado"""
    results = {
        'total_rewards': [],
        'steps': [],
        'final_altitudes': []
    }
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        state = discretization.get_state(obs)
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            action = agent.get_action(state, training=False)
            obs, reward, done, _, info = env.step(np.array([action]))
            next_state = discretization.get_state(obs)
            
            total_reward += reward
            steps += 1
            state = next_state
        
        results['total_rewards'].append(total_reward)
        results['steps'].append(steps)
        results['final_altitudes'].append(info['final_altitude'])
    
    return results

def main():
    """Función principal de demostración"""
    
    print("="*60)
    print("DEMO FLAN - Q-Learning para Control de Descenso de Aeronave")
    print("="*60)
    
    # Configurar entorno
    env = DescentEnv(render_mode=None)
    
    # Definir discretizaciones
    discretizations = [
        SimpleDiscretization("Gruesa", bins=5),
        SimpleDiscretization("Media", bins=10),
        SimpleDiscretization("Fina", bins=15)
    ]
    
    results = {}
    
    for discretization in discretizations:
        print(f"\n{'='*40}")
        print(f"Probando discretización: {discretization.name} ({discretization.bins} bins)")
        print(f"{'='*40}")
        
        # Q-Learning estándar
        print("\n1. Entrenando Q-Learning estándar...")
        ql_agent = SimpleQLearningAgent(discretization, learning_rate=0.1, epsilon=0.2)
        ql_rewards = train_agent(env, ql_agent, discretization, episodes=150, verbose=True)
        
        print("Evaluando Q-Learning estándar...")
        ql_eval = evaluate_agent(env, ql_agent, discretization, num_episodes=30)
        
        # Stochastic Q-Learning
        print("\n2. Entrenando Stochastic Q-Learning...")
        stoch_agent = SimpleStochasticQLearningAgent(discretization, 
                                                    learning_rate=0.1, 
                                                    epsilon=0.2, 
                                                    sample_size=3)
        stoch_rewards = train_agent(env, stoch_agent, discretization, episodes=150, verbose=True)
        
        print("Evaluando Stochastic Q-Learning...")
        stoch_eval = evaluate_agent(env, stoch_agent, discretization, num_episodes=30)
        
        # Guardar resultados
        results[discretization.name] = {
            'qlearning': {
                'training_rewards': ql_rewards,
                'evaluation': ql_eval
            },
            'stochastic': {
                'training_rewards': stoch_rewards,
                'evaluation': stoch_eval
            }
        }
    
    # Generar reporte
    generate_demo_report(results)
    
    return results

def generate_demo_report(results: Dict):
    """Genera un reporte de demostración"""
    
    print("\n" + "="*80)
    print("REPORTE DE DEMOSTRACIÓN - PROYECTO FLAN")
    print("="*80)
    
    # Comparar discretizaciones
    print("\n1. COMPARACIÓN DE DISCRETIZACIONES")
    print("-" * 50)
    
    for discretization_name, discretization_results in results.items():
        print(f"\nDiscretización: {discretization_name}")
        
        ql_avg = np.mean(discretization_results['qlearning']['evaluation']['total_rewards'])
        stoch_avg = np.mean(discretization_results['stochastic']['evaluation']['total_rewards'])
        
        print(f"  Q-Learning estándar: {ql_avg:.2f}")
        print(f"  Stochastic Q-Learning: {stoch_avg:.2f}")
        
        if ql_avg != 0:
            improvement = ((stoch_avg - ql_avg) / abs(ql_avg)) * 100
            print(f"  Mejora: {improvement:.1f}%")
        
        # Métricas adicionales
        ql_steps = np.mean(discretization_results['qlearning']['evaluation']['steps'])
        stoch_steps = np.mean(discretization_results['stochastic']['evaluation']['steps'])
        print(f"  Tiempo promedio Q-Learning: {ql_steps:.1f} pasos")
        print(f"  Tiempo promedio Stochastic: {stoch_steps:.1f} pasos")
    
    # Encontrar mejor configuración
    best_config = None
    best_score = -np.inf
    
    for discretization_name, discretization_results in results.items():
        for agent_type in ['qlearning', 'stochastic']:
            avg_reward = np.mean(discretization_results[agent_type]['evaluation']['total_rewards'])
            if avg_reward > best_score:
                best_score = avg_reward
                best_config = (discretization_name, agent_type)
    
    print(f"\n2. MEJOR CONFIGURACIÓN")
    print("-" * 50)
    print(f"Discretización: {best_config[0]}")
    print(f"Algoritmo: {best_config[1]}")
    print(f"Score promedio: {best_score:.2f}")
    
    # Generar gráficos simples
    generate_demo_plots(results)

def generate_demo_plots(results: Dict):
    """Genera gráficos de demostración"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Comparación de rendimiento por discretización
    discretizations = list(results.keys())
    ql_scores = [np.mean(results[d]['qlearning']['evaluation']['total_rewards']) for d in discretizations]
    stoch_scores = [np.mean(results[d]['stochastic']['evaluation']['total_rewards']) for d in discretizations]
    
    x = np.arange(len(discretizations))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, ql_scores, width, label='Q-Learning Estándar', alpha=0.8)
    axes[0, 0].bar(x + width/2, stoch_scores, width, label='Stochastic Q-Learning', alpha=0.8)
    axes[0, 0].set_xlabel('Discretización')
    axes[0, 0].set_ylabel('Recompensa Promedio')
    axes[0, 0].set_title('Comparación de Rendimiento')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(discretizations)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Curvas de aprendizaje para la mejor discretización
    best_discretization = max(results.keys(), 
                            key=lambda x: np.mean(results[x]['stochastic']['evaluation']['total_rewards']))
    
    ql_rewards = results[best_discretization]['qlearning']['training_rewards']
    stoch_rewards = results[best_discretization]['stochastic']['training_rewards']
    
    # Promedio móvil
    window = 20
    ql_smooth = np.convolve(ql_rewards, np.ones(window)/window, mode='valid')
    stoch_smooth = np.convolve(stoch_rewards, np.ones(window)/window, mode='valid')
    
    axes[0, 1].plot(ql_smooth, label='Q-Learning Estándar', alpha=0.8)
    axes[0, 1].plot(stoch_smooth, label='Stochastic Q-Learning', alpha=0.8)
    axes[0, 1].set_xlabel('Episodio')
    axes[0, 1].set_ylabel('Recompensa Promedio (ventana móvil)')
    axes[0, 1].set_title(f'Curvas de Aprendizaje - {best_discretization}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribución de recompensas finales
    best_results = results[best_discretization]['stochastic']['evaluation']
    
    axes[1, 0].hist(best_results['total_rewards'], bins=15, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Recompensa Total')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].set_title(f'Distribución de Recompensas - {best_discretization}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Comparación de tiempos de supervivencia
    ql_times = results[best_discretization]['qlearning']['evaluation']['steps']
    stoch_times = results[best_discretization]['stochastic']['evaluation']['steps']
    
    axes[1, 1].boxplot([ql_times, stoch_times], labels=['Q-Learning', 'Stochastic'])
    axes[1, 1].set_ylabel('Tiempo de Supervivencia (pasos)')
    axes[1, 1].set_title('Comparación de Tiempos de Supervivencia')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_flan_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nGráficos guardados en 'demo_flan_results.png'")

if __name__ == "__main__":
    results = main() 