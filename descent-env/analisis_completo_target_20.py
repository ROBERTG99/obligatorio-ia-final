#!/usr/bin/env python3
"""
ANÁLISIS COMPLETO PARA ALCANZAR RECOMPENSA -20

Analiza todos los resultados previos y genera la estrategia óptima
para alcanzar consistentemente una recompensa de -20.

Basado en:
- flan_results_10k.json (mejor: -61.08)
- mejoras_radicales_final.py (discretización ultra-fina)
- Paralelización total (todos los cores)
- Ensemble de agentes
- Reward shaping progresivo + agresivo
"""

import numpy as np
import json
import os
import time
import psutil
from contextlib import redirect_stderr, redirect_stdout
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any, Optional
import pickle
import matplotlib.pyplot as plt

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

class AnalisisPrevio:
    """Analiza resultados de experimentos anteriores"""
    
    def __init__(self):
        self.resultados_anteriores = {}
        self.mejores_params = {}
        
    def cargar_resultados(self):
        """Carga y analiza resultados de experimentos previos"""
        archivos = [
            'flan_results_10k.json',
            'flan_results.json', 
            'mejoras_radicales_final_results.json'
        ]
        
        print("📊 ANÁLISIS DE RESULTADOS PREVIOS")
        print("="*50)
        
        for archivo in archivos:
            if os.path.exists(archivo):
                try:
                    with open(archivo, 'r') as f:
                        data = json.load(f)
                    self.resultados_anteriores[archivo] = data
                    print(f"✅ Cargado: {archivo}")
                    self._analizar_archivo(archivo, data)
                except Exception as e:
                    print(f"❌ Error cargando {archivo}: {e}")
        
        return self._extraer_mejores_configs()
    
    def _analizar_archivo(self, archivo: str, data: Dict):
        """Analiza un archivo específico"""
        if 'flan_results' in archivo:
            self._analizar_flan_results(archivo, data)
        elif 'mejoras_radicales' in archivo:
            self._analizar_mejoras_radicales(archivo, data)
    
    def _analizar_flan_results(self, archivo: str, data: Dict):
        """Analiza resultados FLAN"""
        for scheme_name, scheme_data in data.items():
            if isinstance(scheme_data, dict) and 'qlearning' in scheme_data:
                ql_score = scheme_data['qlearning'].get('best_score', -999)
                stoch_score = scheme_data['stochastic'].get('best_score', -999)
                
                print(f"   📈 {scheme_name}:")
                print(f"      • Q-Learning: {ql_score:.2f}")
                print(f"      • Stochastic: {stoch_score:.2f}")
                
                # Guardar mejores parámetros
                if ql_score > -70:
                    self.mejores_params[f"{archivo}_{scheme_name}_ql"] = {
                        'params': scheme_data['qlearning']['best_params'],
                        'score': ql_score,
                        'type': 'qlearning'
                    }
                
                if stoch_score > -70:
                    self.mejores_params[f"{archivo}_{scheme_name}_stoch"] = {
                        'params': scheme_data['stochastic']['best_params'],
                        'score': stoch_score,
                        'type': 'stochastic'
                    }
    
    def _analizar_mejoras_radicales(self, archivo: str, data: Dict):
        """Analiza resultados de mejoras radicales"""
        if 'metricas_finales' in data:
            metrics = data['metricas_finales']
            score = metrics.get('avg_reward', -999)
            print(f"   🚀 Mejoras Radicales: {score:.2f}")
            
            if score > -50:
                self.mejores_params[f"{archivo}_final"] = {
                    'params': data.get('best_params', {}),
                    'score': score,
                    'type': 'mejoras_radicales'
                }
    
    def _extraer_mejores_configs(self) -> Dict:
        """Extrae las mejores configuraciones encontradas"""
        if not self.mejores_params:
            print("⚠️  No se encontraron configuraciones válidas")
            return self._configuracion_default()
        
        # Ordenar por score
        sorted_configs = sorted(self.mejores_params.items(), 
                              key=lambda x: x[1]['score'], reverse=True)
        
        mejor_config = sorted_configs[0][1]
        print(f"\n🏆 MEJOR CONFIGURACIÓN ENCONTRADA:")
        print(f"   • Score: {mejor_config['score']:.2f}")
        print(f"   • Tipo: {mejor_config['type']}")
        print(f"   • Parámetros: {mejor_config['params']}")
        
        return mejor_config
    
    def _configuracion_default(self) -> Dict:
        """Configuración por defecto basada en análisis teórico"""
        return {
            'params': {
                'learning_rate': 0.05,
                'discount_factor': 0.99,
                'epsilon': 0.3,
                'epsilon_decay': 0.9996,
                'use_double_q': True,
                'use_reward_shaping': True
            },
            'score': -65.0,
            'type': 'default'
        }

class DiscretizacionAdaptiva:
    """Discretización que se adapta según el objetivo"""
    
    def __init__(self, objetivo_score: float = -20.0):
        self.objetivo = objetivo_score
        
        # Discretización adaptativa según objetivo
        if objetivo_score >= -25:  # Objetivo ambicioso
            self.altitude_bins = 60
            self.velocity_bins = 60 
            self.target_alt_bins = 60
            self.runway_dist_bins = 60
            self.action_bins = 40
            print("🎯 Discretización ULTRA-FINA para objetivo -20")
        else:  # Objetivo moderado
            self.altitude_bins = 40
            self.velocity_bins = 40
            self.target_alt_bins = 40
            self.runway_dist_bins = 40
            self.action_bins = 25
            print("📊 Discretización FINA para objetivo moderado")
        
        # Espacios optimizados
        self.altitude_space = np.linspace(-3, 3, self.altitude_bins)
        self.velocity_space = np.linspace(-4, 4, self.velocity_bins)
        self.target_alt_space = np.linspace(-3, 3, self.target_alt_bins)
        self.runway_dist_space = np.linspace(-3, 3, self.runway_dist_bins)
        self.action_space = np.linspace(-1, 1, self.action_bins)
        
        total = (self.altitude_bins * self.velocity_bins * 
                self.target_alt_bins * self.runway_dist_bins * 
                self.action_bins)
        print(f"   • Total parámetros: {total:,}")
    
    def get_state(self, obs: Dict) -> Tuple[int, int, int, int]:
        """Convierte observación a estado discreto"""
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
        """Convierte acción continua a índice"""
        action = np.clip(action, -1, 1)
        action_idx = np.digitize(action, self.action_space) - 1
        return int(np.clip(action_idx, 0, self.action_bins - 1))
    
    def get_action_from_index(self, action_idx: int) -> float:
        """Convierte índice a acción continua"""
        return self.action_space[action_idx]

class RewardShaperTarget20:
    """Reward shaping específico para alcanzar -20"""
    
    def __init__(self):
        self.prev_altitude_error = None
        self.prev_action = None
        self.episodio = 0
        
    def shape_reward(self, obs: Dict, action: float, reward: float, done: bool) -> float:
        """Reward shaping optimizado para -20"""
        shaped_reward = reward
        
        current_alt = obs['altitude'][0]
        target_alt = obs['target_altitude'][0]
        runway_dist = obs['runway_distance'][0]
        vz = obs['vz'][0]
        altitude_error = abs(target_alt - current_alt)
        
        # Factor de progreso (más agresivo que antes)
        progress = min(1.0, self.episodio / 15000)
        
        # BONIFICACIONES EXTREMAS para precision
        if altitude_error < 0.005:  # Ultra-preciso
            shaped_reward += 1000.0 * progress
        elif altitude_error < 0.01:
            shaped_reward += 500.0 * progress
        elif altitude_error < 0.02:
            shaped_reward += 200.0 * progress
        elif altitude_error < 0.05:
            shaped_reward += 100.0 * progress
        elif altitude_error < 0.1:
            shaped_reward += 50.0 * progress
        
        # PENALIZACIONES por errores grandes
        if altitude_error > 0.3:
            shaped_reward -= (altitude_error - 0.3) ** 2 * 2000 * progress
        elif altitude_error > 0.2:
            shaped_reward -= (altitude_error - 0.2) ** 2 * 1000 * progress
        
        # Bonificación por mejora consistente
        if self.prev_altitude_error is not None:
            improvement = self.prev_altitude_error - altitude_error
            shaped_reward += improvement * 1000 * progress
        
        # Velocidad vertical óptima
        optimal_vz = -0.1 if current_alt > target_alt else 0.1
        vz_error = abs(vz - optimal_vz)
        shaped_reward += max(0, (0.2 - vz_error) * 100 * progress)
        
        # Suavidad de acciones
        if self.prev_action is not None:
            action_smoothness = abs(action - self.prev_action)
            if action_smoothness < 0.2:
                shaped_reward += (0.2 - action_smoothness) * 100 * progress
        
        # JACKPOT final para aterrizaje perfecto
        if done and runway_dist <= 0:
            if altitude_error < 0.005:
                shaped_reward += 5000.0 * progress  # MEGA JACKPOT
            elif altitude_error < 0.01:
                shaped_reward += 2500.0 * progress
            elif altitude_error < 0.02:
                shaped_reward += 1000.0 * progress
            elif altitude_error < 0.05:
                shaped_reward += 500.0 * progress
        
        self.prev_altitude_error = altitude_error
        self.prev_action = action
        
        return shaped_reward
    
    def reset(self):
        """Reset para nuevo episodio"""
        self.episodio += 1
        self.prev_altitude_error = None
        self.prev_action = None

class AgenteOptimizado:
    """Agente Q-Learning optimizado para -20"""
    
    def __init__(self, discretization: DiscretizacionAdaptiva, **params):
        self.discretization = discretization
        self.learning_rate = params.get('learning_rate', 0.05)
        self.discount_factor = params.get('discount_factor', 0.99)
        self.epsilon = params.get('epsilon', 0.3)
        self.epsilon_min = params.get('epsilon_min', 0.01)
        self.epsilon_decay = params.get('epsilon_decay', 0.9996)
        self.use_double_q = params.get('use_double_q', True)
        
        # Tabla Q (doble si se especifica)
        shape = (discretization.altitude_bins, discretization.velocity_bins,
                discretization.target_alt_bins, discretization.runway_dist_bins,
                discretization.action_bins)
        
        self.Q1 = np.zeros(shape)
        self.Q2 = np.zeros(shape) if self.use_double_q else None
        self.visits = np.zeros(shape)
        
        # Reward shaper específico para -20
        self.reward_shaper = RewardShaperTarget20()
        
        self.episodios = 0
        
    def get_action(self, state: Tuple[int, int, int, int], training: bool = True) -> float:
        """Selección de acción epsilon-greedy mejorada"""
        if training and np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, self.discretization.action_bins)
        else:
            if self.use_double_q and self.Q2 is not None:
                q_values = (self.Q1[state] + self.Q2[state]) / 2
            else:
                q_values = self.Q1[state]
            action_idx = int(np.argmax(q_values))
        
        return self.discretization.get_action_from_index(action_idx)
    
    def update(self, state, action, reward, next_state, done, obs):
        """Actualización Double Q-Learning con reward shaping"""
        action_idx = self.discretization.get_action_index(action)
        
        # Reward shaping agresivo
        reward = self.reward_shaper.shape_reward(obs, action, reward, done)
        
        # Learning rate adaptativo
        self.visits[state][action_idx] += 1
        alpha = self.learning_rate / (1 + 0.0001 * self.visits[state][action_idx])
        
        # Double Q-Learning
        if self.use_double_q and self.Q2 is not None:
            if np.random.random() < 0.5:
                current_q = self.Q1[state][action_idx]
                if done:
                    target_q = reward
                else:
                    best_action_idx = int(np.argmax(self.Q1[next_state]))
                    max_next_q = self.Q2[next_state][best_action_idx]
                    target_q = reward + self.discount_factor * max_next_q
                self.Q1[state][action_idx] = current_q + alpha * (target_q - current_q)
            else:
                current_q = self.Q2[state][action_idx]
                if done:
                    target_q = reward
                else:
                    best_action_idx = int(np.argmax(self.Q2[next_state]))
                    max_next_q = self.Q1[next_state][best_action_idx]
                    target_q = reward + self.discount_factor * max_next_q
                self.Q2[state][action_idx] = current_q + alpha * (target_q - current_q)
        else:
            # Q-Learning estándar
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
    
    def finish_episode(self, reward):
        """Finalizar episodio"""
        self.episodios += 1
        self.reward_shaper.reset()

def entrenar_y_evaluar_combo(combo_params):
    """Función para entrenar y evaluar una combinación en paralelo"""
    params, discretization_config, objetivo = combo_params
    
    # Silenciar salida
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            env = DescentEnv(render_mode=None)
    
    # Recrear discretización
    discretization = DiscretizacionAdaptiva(objetivo)
    agent = AgenteOptimizado(discretization, **params)
    
    # Entrenamiento intensivo (5000 episodios para objetivo ambicioso)
    TRAINING_EPISODES = 5000 if objetivo >= -25 else 3000
    
    for _ in range(TRAINING_EPISODES):
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
        
        agent.finish_episode(total_reward)
    
    # Evaluación extensiva
    EVAL_EPISODES = 500
    eval_rewards = []
    
    for _ in range(EVAL_EPISODES):
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
    
    # Métricas
    score = float(np.mean(eval_rewards))
    std = float(np.std(eval_rewards))
    best = float(np.max(eval_rewards))
    success_rate = float(np.sum(np.array(eval_rewards) >= objetivo) / len(eval_rewards))
    
    return {
        'params': params,
        'score': score,
        'std': std,
        'best': best,
        'success_rate': success_rate,
        'eval_rewards': eval_rewards,
        'objetivo_alcanzado': score >= objetivo
    }

def busqueda_hiperparametros_target_20(mejor_config_previa):
    """Búsqueda optimizada específicamente para alcanzar -20"""
    
    print("\n🎯 BÚSQUEDA HIPERPARÁMETROS PARA OBJETIVO -20")
    print("="*60)
    
    # Grid extendido basado en mejores configuraciones previas
    base_lr = mejor_config_previa['params'].get('learning_rate', 0.05)
    base_gamma = mejor_config_previa['params'].get('discount_factor', 0.99)
    
    param_grid = [
        # Configuraciones conservadoras (basadas en análisis previo)
        {'learning_rate': base_lr * 0.8, 'discount_factor': base_gamma, 'epsilon': 0.4, 'epsilon_decay': 0.9995},
        {'learning_rate': base_lr, 'discount_factor': base_gamma, 'epsilon': 0.35, 'epsilon_decay': 0.9996},
        {'learning_rate': base_lr * 1.2, 'discount_factor': base_gamma, 'epsilon': 0.3, 'epsilon_decay': 0.9997},
        
        # Configuraciones agresivas para -20
        {'learning_rate': 0.08, 'discount_factor': 0.995, 'epsilon': 0.5, 'epsilon_decay': 0.9994},
        {'learning_rate': 0.06, 'discount_factor': 0.997, 'epsilon': 0.4, 'epsilon_decay': 0.9995},
        {'learning_rate': 0.04, 'discount_factor': 0.999, 'epsilon': 0.35, 'epsilon_decay': 0.9996},
        
        # Configuraciones experimentales
        {'learning_rate': 0.1, 'discount_factor': 0.992, 'epsilon': 0.6, 'epsilon_decay': 0.9993},
        {'learning_rate': 0.03, 'discount_factor': 0.998, 'epsilon': 0.25, 'epsilon_decay': 0.9998},
        {'learning_rate': 0.12, 'discount_factor': 0.99, 'epsilon': 0.45, 'epsilon_decay': 0.9994},
        
        # Configuraciones híbridas
        {'learning_rate': 0.075, 'discount_factor': 0.996, 'epsilon': 0.4, 'epsilon_decay': 0.9995},
        {'learning_rate': 0.055, 'discount_factor': 0.994, 'epsilon': 0.38, 'epsilon_decay': 0.9996},
        {'learning_rate': 0.09, 'discount_factor': 0.993, 'epsilon': 0.42, 'epsilon_decay': 0.9994}
    ]
    
    # Añadir parámetros fijos
    for params in param_grid:
        params.update({
            'use_double_q': True,
            'use_reward_shaping': True,
            'epsilon_min': 0.01
        })
    
    print(f"🔍 Evaluando {len(param_grid)} configuraciones con paralelización total")
    
    # Paralelización total
    cpu_count = psutil.cpu_count(logical=True) or os.cpu_count()
    num_cores = cpu_count if cpu_count and cpu_count > 0 else 2
    print(f"⚡ Usando {num_cores} cores para búsqueda masiva")
    
    objetivo = -20.0
    discretization_config = {}  # Config para recrear discretización
    
    # Preparar tareas
    tasks = [(params, discretization_config, objetivo) for params in param_grid]
    
    results = []
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        future_to_params = {
            executor.submit(entrenar_y_evaluar_combo, task): task[0] for task in tasks
        }
        
        for i, future in enumerate(as_completed(future_to_params), 1):
            try:
                result = future.result()
                results.append(result)
                
                objetivo_check = "✅" if result['objetivo_alcanzado'] else "❌"
                print(f"   [{i:2d}/{len(param_grid)}] {objetivo_check} Score: {result['score']:6.2f} "
                      f"(Best: {result['best']:6.2f}, Success: {result['success_rate']*100:4.1f}%) "
                      f"LR: {result['params']['learning_rate']:.3f}")
                
            except Exception as e:
                print(f"❌ Error en configuración {i}: {e}")
    
    # Análisis de resultados
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n📊 ANÁLISIS DE RESULTADOS:")
    print(f"   • Configuraciones que alcanzaron -20: {sum(1 for r in results if r['objetivo_alcanzado'])}")
    print(f"   • Mejor score: {results[0]['score']:.2f}")
    print(f"   • Configuraciones con score > -25: {sum(1 for r in results if r['score'] > -25)}")
    
    return results

def entrenamiento_final_target_20(mejores_configs):
    """Entrenamiento final intensivo con las mejores configuraciones"""
    
    print(f"\n🚀 ENTRENAMIENTO FINAL PARA ALCANZAR -20")
    print("="*50)
    
    # Seleccionar top 3 configuraciones
    top_configs = mejores_configs[:3]
    
    print(f"🏆 Entrenando TOP-3 configuraciones con 15,000 episodios cada una:")
    for i, config in enumerate(top_configs, 1):
        print(f"   {i}. Score: {config['score']:.2f}, Params: {config['params']}")
    
    # Entrenamiento intensivo de cada configuración
    agentes_finales = []
    
    for i, config in enumerate(top_configs):
        print(f"\n🎯 Entrenando configuración {i+1}/3...")
        
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                env = DescentEnv(render_mode=None)
        
        discretization = DiscretizacionAdaptiva(-20.0)
        agent = AgenteOptimizado(discretization, **config['params'])
        
        # Entrenamiento MASIVO
        FINAL_EPISODES = 15000
        training_rewards = []
        
        for episode in range(FINAL_EPISODES):
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
            
            agent.finish_episode(total_reward)
            training_rewards.append(total_reward)
            
            # Progreso cada 2500 episodios
            if (episode + 1) % 2500 == 0:
                recent_avg = np.mean(training_rewards[-1000:])
                recent_best = np.max(training_rewards[-1000:])
                success_rate = np.sum(np.array(training_rewards[-1000:]) >= -20) / 1000 * 100
                print(f"      Episode {episode+1}: Avg {recent_avg:.2f}, Best {recent_best:.2f}, Success {success_rate:.1f}%")
        
        # Evaluación final robusta
        print(f"   📊 Evaluación final robusta (1000 episodios)...")
        eval_rewards = []
        
        for _ in range(1000):
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
        
        final_score = np.mean(eval_rewards)
        final_std = np.std(eval_rewards)
        final_best = np.max(eval_rewards)
        success_rate = np.sum(np.array(eval_rewards) >= -20) / len(eval_rewards) * 100
        
        agentes_finales.append({
            'agent': agent,
            'discretization': discretization,
            'config': config,
            'final_score': final_score,
            'final_std': final_std,
            'final_best': final_best,
            'success_rate': success_rate,
            'training_rewards': training_rewards,
            'eval_rewards': eval_rewards,
            'objetivo_alcanzado': final_score >= -20
        })
        
        print(f"   ✅ Resultado: {final_score:.2f} ± {final_std:.2f} (Mejor: {final_best:.2f}, Éxito: {success_rate:.1f}%)")
        
        if final_score >= -20:
            print(f"   🎉 ¡OBJETIVO -20 ALCANZADO!")
    
    return agentes_finales

def generar_reporte_final(analisis_previo, resultados_busqueda, agentes_finales):
    """Genera reporte completo del análisis"""
    
    print(f"\n{'='*80}")
    print("📋 REPORTE FINAL - ANÁLISIS COMPLETO PARA ALCANZAR -20")
    print('='*80)
    
    # Mejor agente final
    agentes_finales.sort(key=lambda x: x['final_score'], reverse=True)
    mejor_agente = agentes_finales[0]
    
    print(f"\n🏆 MEJOR AGENTE ENTRENADO:")
    print(f"   • Score final: {mejor_agente['final_score']:.2f} ± {mejor_agente['final_std']:.2f}")
    print(f"   • Mejor episodio: {mejor_agente['final_best']:.2f}")
    print(f"   • Tasa de éxito (-20): {mejor_agente['success_rate']:.1f}%")
    print(f"   • Objetivo alcanzado: {'✅ SÍ' if mejor_agente['objetivo_alcanzado'] else '❌ NO'}")
    
    print(f"\n📊 PARÁMETROS ÓPTIMOS ENCONTRADOS:")
    for param, valor in mejor_agente['config']['params'].items():
        print(f"   • {param}: {valor}")
    
    print(f"\n📈 PROGRESO DESDE RESULTADOS ANTERIORES:")
    mejor_anterior = max(analisis_previo.mejores_params.values(), key=lambda x: x['score'])
    mejora = mejor_agente['final_score'] - mejor_anterior['score']
    mejora_pct = (mejora / abs(mejor_anterior['score'])) * 100
    
    print(f"   • Score anterior: {mejor_anterior['score']:.2f}")
    print(f"   • Score actual: {mejor_agente['final_score']:.2f}")
    print(f"   • Mejora absoluta: +{mejora:.2f} puntos")
    print(f"   • Mejora porcentual: +{mejora_pct:.1f}%")
    
    print(f"\n🎯 ANÁLISIS DE ÉXITO:")
    if mejor_agente['objetivo_alcanzado']:
        print(f"   ✅ OBJETIVO -20 ALCANZADO EXITOSAMENTE")
        print(f"   🎉 Score promedio: {mejor_agente['final_score']:.2f} >= -20")
        print(f"   📊 Consistencia: {mejor_agente['success_rate']:.1f}% de episodios exitosos")
        print(f"   🏆 Mejor episodio individual: {mejor_agente['final_best']:.2f}")
    else:
        falta = -20 - mejor_agente['final_score']
        print(f"   ⚠️  Objetivo -20 no alcanzado por {falta:.2f} puntos")
        print(f"   📈 Progreso significativo: {mejora:.2f} puntos de mejora")
        
        if mejor_agente['final_best'] >= -20:
            print(f"   💡 El agente SÍ puede alcanzar -20 (mejor: {mejor_agente['final_best']:.2f})")
            print(f"   🔧 Problema: CONSISTENCIA, no capacidad")
        else:
            print(f"   🔧 Necesario: Más entrenamiento o ajuste de algoritmo")
    
    # Estadísticas de la búsqueda
    exitosos = [r for r in resultados_busqueda if r['objetivo_alcanzado']]
    print(f"\n🔍 ESTADÍSTICAS DE BÚSQUEDA:")
    print(f"   • Configuraciones evaluadas: {len(resultados_busqueda)}")
    print(f"   • Configuraciones exitosas (-20): {len(exitosos)}")
    print(f"   • Tasa de éxito en búsqueda: {len(exitosos)/len(resultados_busqueda)*100:.1f}%")
    
    # Recomendaciones
    print(f"\n💡 RECOMENDACIONES:")
    if mejor_agente['objetivo_alcanzado']:
        print(f"   ✅ Modelo listo para producción")
        print(f"   📋 Documentar configuración óptima")
        print(f"   🔬 Analizar factores de éxito para futuros proyectos")
    elif mejor_agente['final_score'] > -25:
        print(f"   🎯 Muy cerca del objetivo - continuar optimización")
        print(f"   🔧 Ajustar reward shaping o aumentar episodios de entrenamiento")
        print(f"   📊 Considerar ensemble de agentes")
    else:
        print(f"   🧠 Considerar algoritmos más avanzados (DDPG, TD3, SAC)")
        print(f"   🔄 Q-Learning tabular puede haber alcanzado su límite")
        print(f"   📚 Investigar métodos de aproximación de funciones")
    
    # Guardar resultados completos
    reporte = {
        'objetivo': -20.0,
        'mejor_agente': {
            'score': mejor_agente['final_score'],
            'std': mejor_agente['final_std'],
            'best_episode': mejor_agente['final_best'],
            'success_rate': mejor_agente['success_rate'],
            'objetivo_alcanzado': mejor_agente['objetivo_alcanzado'],
            'params': mejor_agente['config']['params']
        },
        'progreso': {
            'score_anterior': mejor_anterior['score'],
            'score_actual': mejor_agente['final_score'],
            'mejora_absoluta': mejora,
            'mejora_porcentual': mejora_pct
        },
        'busqueda_stats': {
            'total_configs': len(resultados_busqueda),
            'configs_exitosas': len(exitosos),
            'tasa_exito_busqueda': len(exitosos)/len(resultados_busqueda)*100
        },
        'todos_los_agentes': [{
            'score': a['final_score'],
            'params': a['config']['params'],
            'objetivo_alcanzado': a['objetivo_alcanzado']
        } for a in agentes_finales]
    }
    
    with open('analisis_completo_target_20_results.json', 'w') as f:
        json.dump(reporte, f, indent=2)
    
    # Guardar mejor modelo
    if mejor_agente['objetivo_alcanzado'] or mejor_agente['final_score'] > -25:
        with open('mejor_modelo_target_20.pkl', 'wb') as f:
            pickle.dump({
                'agent': mejor_agente['agent'],
                'discretization': mejor_agente['discretization'],
                'params': mejor_agente['config']['params'],
                'performance': mejor_agente['final_score'],
                'success_rate': mejor_agente['success_rate']
            }, f)
        print(f"\n💾 Mejor modelo guardado en 'mejor_modelo_target_20.pkl'")
    
    print(f"\n💾 Reporte completo guardado en 'analisis_completo_target_20_results.json'")
    
    return reporte

def main():
    """Función principal del análisis completo"""
    
    print("🎯 ANÁLISIS COMPLETO PARA ALCANZAR RECOMPENSA -20")
    print("="*60)
    print("📊 Analizando resultados previos y generando estrategia óptima")
    print("⚡ Paralelización total habilitada")
    print("🎯 Objetivo: Recompensa consistente >= -20")
    
    start_time = time.time()
    
    # FASE 1: Análisis de resultados previos
    print(f"\n{'='*60}")
    print("FASE 1: ANÁLISIS DE RESULTADOS PREVIOS")
    print('='*60)
    
    analisis = AnalisisPrevio()
    mejor_config_previa = analisis.cargar_resultados()
    
    # FASE 2: Búsqueda optimizada de hiperparámetros
    print(f"\n{'='*60}")
    print("FASE 2: BÚSQUEDA HIPERPARÁMETROS OPTIMIZADA")
    print('='*60)
    
    resultados_busqueda = busqueda_hiperparametros_target_20(mejor_config_previa)
    
    # FASE 3: Entrenamiento final intensivo
    print(f"\n{'='*60}")
    print("FASE 3: ENTRENAMIENTO FINAL INTENSIVO")
    print('='*60)
    
    agentes_finales = entrenamiento_final_target_20(resultados_busqueda)
    
    # FASE 4: Reporte final
    print(f"\n{'='*60}")
    print("FASE 4: GENERACIÓN DE REPORTE FINAL")
    print('='*60)
    
    reporte = generar_reporte_final(analisis, resultados_busqueda, agentes_finales)
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  TIEMPO TOTAL: {elapsed/3600:.1f} horas ({elapsed/60:.0f} minutos)")
    
    print(f"\n🏁 ANÁLISIS COMPLETO FINALIZADO")
    
    return reporte

if __name__ == "__main__":
    results = main() 