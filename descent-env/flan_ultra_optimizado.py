#!/usr/bin/env python3
"""
FLAN Q-LEARNING ULTRA-OPTIMIZADO PARA ALCANZAR -30

Implementa las mejoras de alta prioridad identificadas en el análisis avanzado:
1. Discretización ultra-fina (50×50×50×50×50)
2. Hiperparámetros balanceados (LR: 0.01-0.1)  
3. Reward shaping gradual y estable
4. Análisis de trayectorias en tiempo real

OBJETIVO: Alcanzar recompensa -30 de forma CONSISTENTE
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
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil

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
        
        print(f"🔧 Discretización Ultra-Fina inicializada:")
        print(f"   • Estados: {self.altitude_bins}×{self.velocity_bins}×{self.target_alt_bins}×{self.runway_dist_bins}")
        print(f"   • Acciones: {self.action_bins}")
        print(f"   • Total combinaciones: {self.altitude_bins * self.velocity_bins * self.target_alt_bins * self.runway_dist_bins * self.action_bins:,}")
        
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

class RewardShaperGradual:
    """Reward shaping gradual para estabilidad y convergencia"""
    
    def __init__(self):
        self.prev_altitude_error = None
        self.prev_altitude = None
        self.prev_action = None
        self.episode_count = 0
        self.steps = 0
        
    def shape_reward(self, obs: Dict, action: float, reward: float, done: bool) -> float:
        """Reward shaping gradual que mejora con el tiempo"""
        shaped_reward = reward
        
        # Obtener valores actuales
        current_alt = obs['altitude'][0]
        target_alt = obs['target_altitude'][0]
        runway_dist = obs['runway_distance'][0]
        vz = obs['vz'][0]
        
        altitude_error = abs(target_alt - current_alt)
        
        # MEJORA CLAVE: Bonificación gradual que aumenta con episodios
        # Comienza conservador y se vuelve más agresivo con el tiempo
        bonus_scale = min(1.0, self.episode_count / 20000)  # Más gradual
        
        # Bonificaciones escaladas por precisión
        if altitude_error < 0.01:
            shaped_reward += 100.0 * bonus_scale  # vs 2000.0 anterior
        elif altitude_error < 0.02:
            shaped_reward += 50.0 * bonus_scale   # vs 500.0 anterior
        elif altitude_error < 0.05:
            shaped_reward += 25.0 * bonus_scale   # vs 200.0 anterior
        elif altitude_error < 0.1:
            shaped_reward += 10.0 * bonus_scale   # vs 100.0 anterior
        elif altitude_error < 0.2:
            shaped_reward += 5.0 * bonus_scale    # vs 25.0 anterior
            
        # Penalización gradual por errores grandes
        if altitude_error > 0.5:
            penalty_scale = min(1.0, self.episode_count / 10000)
            shaped_reward -= (altitude_error - 0.5) ** 2 * 20 * penalty_scale
        
        # Bonificación por mejora (más estable)
        if self.prev_altitude_error is not None:
            improvement = self.prev_altitude_error - altitude_error
            shaped_reward += improvement * 10 * bonus_scale  # vs 500 anterior
            
        # Bonificación por suavidad de acciones
        if self.prev_action is not None:
            action_smoothness = abs(action - self.prev_action)
            if action_smoothness < 0.2:
                shaped_reward += (0.2 - action_smoothness) * 5 * bonus_scale
        
        # Velocidad vertical apropiada (más gradual)
        if runway_dist > 0.5:
            optimal_vz = -0.2 if current_alt > target_alt else 0.2
        else:
            optimal_vz = -0.1 if current_alt > target_alt else 0.1
            
        vz_error = abs(vz - optimal_vz)
        shaped_reward += max(0, (0.5 - vz_error) * 10 * bonus_scale)
        
        # Jackpot final más moderado
        if done and runway_dist <= 0:
            if altitude_error < 0.01:
                shaped_reward += 200.0 * bonus_scale  # vs 2000.0
            elif altitude_error < 0.02:
                shaped_reward += 100.0 * bonus_scale  # vs 1000.0
            elif altitude_error < 0.05:
                shaped_reward += 50.0 * bonus_scale   # vs 500.0
            elif altitude_error < 0.1:
                shaped_reward += 25.0 * bonus_scale   # vs 250.0
        
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

class QLearningAgentBalanceado:
    """Q-Learning con hiperparámetros balanceados para estabilidad"""
    
    def __init__(self, discretization: DiscretizationUltraFina,
                 learning_rate: float = 0.05,
                 discount_factor: float = 0.99,
                 epsilon: float = 0.2,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.9995,
                 use_reward_shaping: bool = True):
        
        self.discretization = discretization
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.use_reward_shaping = use_reward_shaping
        
        # Tabla Q para discretización ultra-fina
        shape = (discretization.altitude_bins,
                discretization.velocity_bins,
                discretization.target_alt_bins,
                discretization.runway_dist_bins,
                discretization.action_bins)
        
        self.Q = np.zeros(shape)
        self.visits = np.zeros(shape)
        
        # Reward shaper gradual
        self.reward_shaper = RewardShaperGradual() if use_reward_shaping else None
        
        print(f"🤖 QLearningAgentBalanceado inicializado:")
        print(f"   • Learning rate: {learning_rate}")
        print(f"   • Discount factor: {discount_factor}")  
        print(f"   • Epsilon inicial: {epsilon}")
        print(f"   • Tabla Q shape: {shape}")
        print(f"   • Memoria requerida: ~{np.prod(shape) * 8 / 1024**3:.2f} GB")
        
    def get_action(self, state: Tuple[int, int, int, int], training: bool = True) -> float:
        """Selección de acción balanceada"""
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
               done: bool, obs: Optional[Dict] = None):
        """Actualización Q-Learning balanceada"""
        
        action_idx = self.discretization.get_action_index(action)
        
        # Aplicar reward shaping gradual
        if self.reward_shaper is not None and obs is not None:
            reward = self.reward_shaper.shape_reward(obs, action, reward, done)
        
        # Learning rate adaptativo más estable
        self.visits[state][action_idx] += 1
        alpha = self.learning_rate / (1 + 0.01 * self.visits[state][action_idx])  # Más gradual
        
        # Actualización Q-Learning estándar
        current_q = self.Q[state][action_idx]
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.Q[next_state])
            target_q = reward + self.discount_factor * max_next_q
        
        self.Q[state][action_idx] = current_q + alpha * (target_q - current_q)
        
        # Decay epsilon más gradual
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def reset_episode(self):
        """Reset para nuevo episodio"""
        if self.reward_shaper is not None:
            self.reward_shaper.reset()

class AnalizadorTrayectorias:
    """Analiza trayectorias para identificar patrones de éxito"""
    
    def __init__(self):
        self.trayectorias_exitosas = []  # Recompensa >= -35
        self.trayectorias_fallidas = []   # Recompensa < -60
        self.episodio_actual = []
        
    def agregar_paso(self, state, action, reward, next_state, done, obs):
        """Agrega un paso a la trayectoria actual"""
        self.episodio_actual.append({
            'state': state,
            'action': action, 
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'obs': obs.copy()
        })
        
        if done:
            total_reward = sum(paso['reward'] for paso in self.episodio_actual)
            
            if total_reward >= -35:
                self.trayectorias_exitosas.append(self.episodio_actual.copy())
                if len(self.trayectorias_exitosas) > 10:  # Mantener solo las mejores
                    self.trayectorias_exitosas.pop(0)
            elif total_reward < -60:
                self.trayectorias_fallidas.append(self.episodio_actual.copy())
                if len(self.trayectorias_fallidas) > 10:
                    self.trayectorias_fallidas.pop(0)
            
            self.episodio_actual = []
    
    def analizar_patrones(self):
        """Analiza patrones en trayectorias exitosas vs fallidas"""
        if not self.trayectorias_exitosas or not self.trayectorias_fallidas:
            return
        
        print(f"\n📊 ANÁLISIS DE TRAYECTORIAS:")
        print(f"   • Exitosas: {len(self.trayectorias_exitosas)}")
        print(f"   • Fallidas: {len(self.trayectorias_fallidas)}")
        
        # Analizar acciones promedio
        acciones_exitosas = []
        acciones_fallidas = []
        
        for trayectoria in self.trayectorias_exitosas:
            acciones_exitosas.extend([paso['action'] for paso in trayectoria])
        
        for trayectoria in self.trayectorias_fallidas:
            acciones_fallidas.extend([paso['action'] for paso in trayectoria])
        
        if acciones_exitosas and acciones_fallidas:
            print(f"   • Acción promedio exitosa: {np.mean(acciones_exitosas):.3f}")
            print(f"   • Acción promedio fallida: {np.mean(acciones_fallidas):.3f}")
            print(f"   • Diferencia: {np.mean(acciones_exitosas) - np.mean(acciones_fallidas):.3f}")

def entrenar_agente_ultra_optimizado(episodes: int = 50000) -> QLearningAgentBalanceado:
    """Entrenamiento ultra-optimizado con todas las mejoras"""
    
    print("\n" + "="*80)
    print("🚀 ENTRENAMIENTO ULTRA-OPTIMIZADO INICIADO")
    print("="*80)
    print(f"📊 CONFIGURACIÓN:")
    print(f"   • Episodios: {episodes:,}")
    print(f"   • Discretización: Ultra-fina (50×50×50×50×50)")
    print(f"   • Reward shaping: Gradual y estable")
    print(f"   • Hiperparámetros: Balanceados")
    print(f"   • Tiempo estimado: 4-8 horas")
    
    # Crear entorno 
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            env = DescentEnv(render_mode=None)
    
    # Crear componentes ultra-optimizados
    discretization = DiscretizationUltraFina()
    agent = QLearningAgentBalanceado(discretization)
    analizador = AnalizadorTrayectorias()
    
    # Variables de seguimiento
    episode_rewards = []
    mejores_episodios = []
    convergencia_window = 500
    
    print(f"\n🎯 INICIANDO ENTRENAMIENTO...")
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
            
            # Actualizar agente
            agent.update(state, action, reward, next_state, done, next_obs)
            
            # Agregar a analizador de trayectorias
            analizador.agregar_paso(state, action, reward, next_state, done, next_obs)
            
            total_reward += reward
            state = next_state
            step += 1
        
        episode_rewards.append(total_reward)
        
        # Rastrear mejores episodios
        if total_reward >= -35:
            mejores_episodios.append((episode, total_reward))
            
        # Reportar progreso
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(episode_rewards[-convergencia_window:])
            best_recent = np.max(episode_rewards[-convergencia_window:])
            epsilon = agent.epsilon
            elapsed = time.time() - start_time
            
            print(f"📈 Episodio {episode + 1:,}:")
            print(f"   • Promedio reciente: {avg_reward:.2f}")
            print(f"   • Mejor reciente: {best_recent:.2f}")
            print(f"   • Epsilon: {epsilon:.4f}")
            print(f"   • Episodios >= -35: {len(mejores_episodios)}")
            print(f"   • Tiempo: {elapsed/3600:.1f}h")
            
            # Análisis de trayectorias cada 5000 episodios
            if (episode + 1) % 5000 == 0:
                analizador.analizar_patrones()
        
        # Early stopping si conseguimos consistencia
        if len(episode_rewards) >= convergencia_window:
            recent_rewards = episode_rewards[-convergencia_window:]
            if np.mean(recent_rewards) >= -30 and np.percentile(recent_rewards, 25) >= -35:
                print(f"\n🎉 ¡CONVERGENCIA EXITOSA EN EPISODIO {episode + 1}!")
                print(f"   • Promedio últimos {convergencia_window}: {np.mean(recent_rewards):.2f}")
                print(f"   • Percentil 25: {np.percentile(recent_rewards, 25):.2f}")
                break
    
    elapsed_total = time.time() - start_time
    print(f"\n✅ ENTRENAMIENTO COMPLETADO")
    print(f"   • Tiempo total: {elapsed_total/3600:.2f} horas")
    print(f"   • Episodios >= -35: {len(mejores_episodios)}")
    print(f"   • Mejor episodio: {np.max(episode_rewards):.2f}")
    
    return agent, episode_rewards, mejores_episodios

def evaluar_agente_ultra(agent: QLearningAgentBalanceado, 
                        discretization: DiscretizationUltraFina,
                        episodes: int = 1000) -> Dict:
    """Evaluación exhaustiva del agente ultra-optimizado"""
    
    print(f"\n🔍 EVALUACIÓN ULTRA-EXHAUSTIVA ({episodes} episodios)...")
    
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            env = DescentEnv(render_mode=None)
    
    results = {
        'total_rewards': [],
        'steps': [],
        'altitude_errors': [],
        'final_altitudes': [],
        'successful_episodes': 0
    }
    
    for episode in range(episodes):
        obs, _ = env.reset()
        state = discretization.get_state(obs)
        
        total_reward = 0
        steps = 0
        target_altitude = obs['target_altitude'][0]
        
        done = False
        while not done and steps < 500:
            action = agent.get_action(state, training=False)
            obs, reward, done, _, info = env.step(np.array([action]))
            next_state = discretization.get_state(obs)
            
            total_reward += reward
            state = next_state
            steps += 1
        
        results['total_rewards'].append(total_reward)
        results['steps'].append(steps)
        
        final_altitude = info['final_altitude']
        altitude_error = abs(final_altitude - target_altitude)
        results['altitude_errors'].append(altitude_error)
        results['final_altitudes'].append(final_altitude)
        
        if total_reward >= -30:
            results['successful_episodes'] += 1
            
        if (episode + 1) % 200 == 0:
            success_rate = results['successful_episodes'] / (episode + 1) * 100
            avg_reward = np.mean(results['total_rewards'])
            print(f"   Episodio {episode + 1}: Promedio {avg_reward:.2f}, Éxito {success_rate:.1f}%")
    
    return results

def main():
    """Función principal del experimento ultra-optimizado"""
    
    print("🎯 FLAN Q-LEARNING ULTRA-OPTIMIZADO PARA ALCANZAR -30")
    print("="*60)
    
    # FASE 1: Entrenamiento ultra-optimizado
    agent, training_rewards, mejores_episodios = entrenar_agente_ultra_optimizado(episodes=50000)
    
    # FASE 2: Evaluación exhaustiva
    discretization = DiscretizationUltraFina()
    evaluation_results = evaluar_agente_ultra(agent, discretization, episodes=1000)
    
    # FASE 3: Análisis de resultados
    print("\n" + "="*80)
    print("📊 RESULTADOS FINALES ULTRA-OPTIMIZADOS")
    print("="*80)
    
    eval_rewards = evaluation_results['total_rewards']
    success_rate = evaluation_results['successful_episodes'] / len(eval_rewards) * 100
    
    print(f"🎯 MÉTRICAS CLAVE:")
    print(f"   • Recompensa promedio: {np.mean(eval_rewards):.2f}")
    print(f"   • Mejor episodio: {np.max(eval_rewards):.2f}")
    print(f"   • Peor episodio: {np.min(eval_rewards):.2f}")
    print(f"   • Desviación estándar: {np.std(eval_rewards):.2f}")
    print(f"   • Percentil 90: {np.percentile(eval_rewards, 90):.2f}")
    print(f"   • Percentil 75: {np.percentile(eval_rewards, 75):.2f}")
    print(f"   • Percentil 50: {np.percentile(eval_rewards, 50):.2f}")
    
    print(f"\n🏆 ÉXITO:")
    print(f"   • Episodios >= -30: {evaluation_results['successful_episodes']}/1000")
    print(f"   • Tasa de éxito: {success_rate:.1f}%")
    print(f"   • Episodios durante entrenamiento >= -35: {len(mejores_episodios)}")
    
    objetivo_alcanzado = np.mean(eval_rewards) >= -30
    print(f"\n🎯 OBJETIVO ALCANZADO: {'✅ SÍ' if objetivo_alcanzado else '❌ NO'}")
    
    if objetivo_alcanzado:
        print(f"🎉 ¡ÉXITO! Recompensa promedio de {np.mean(eval_rewards):.2f} >= -30")
    else:
        falta = -30 - np.mean(eval_rewards)
        print(f"⚠️  Falta {falta:.2f} puntos para alcanzar -30")
    
    # Guardar resultados
    results_ultra = {
        'training_rewards': training_rewards,
        'evaluation_results': evaluation_results,
        'mejores_episodios': mejores_episodios,
        'success_rate': success_rate,
        'objetivo_alcanzado': objetivo_alcanzado
    }
    
    with open('flan_ultra_optimizado_results.json', 'w') as f:
        # Convertir numpy arrays a listas
        serializable_results = {}
        for key, value in results_ultra.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.float64):
                serializable_results[key] = [float(x) for x in value]
            else:
                serializable_results[key] = value
        
        # Convertir evaluation_results
        eval_serializable = {}
        for key, value in evaluation_results.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], (np.float64, np.int64)):
                eval_serializable[key] = [float(x) for x in value]
            else:
                eval_serializable[key] = value
        serializable_results['evaluation_results'] = eval_serializable
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n💾 Resultados guardados en 'flan_ultra_optimizado_results.json'")
    
    return results_ultra

if __name__ == "__main__":
    results = main() 